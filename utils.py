import re

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA

from yt_dlp import YoutubeDL




def _extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def _get_video_metadata(url: str) -> dict:
    """Get video metadata using yt-dlp (more reliable than pytube)."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title', 'Unknown'),
            'description': info.get('description', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'thumbnail_url': info.get('thumbnail', 'Unknown'),
            'publish_date': info.get('upload_date', 'Unknown'),
            'length': info.get('duration', 0),
            'author': info.get('uploader', 'Unknown'),
            'channel_id': info.get('channel_id', 'Unknown'),
            'webpage_url': info.get('webpage_url', url),
        }


def get_youtube_details(url: str, include_video_info: bool = True):
    """
    Fetch YouTube video transcript and metadata.
    
    Uses YoutubeLoader for transcripts (reliable) and yt-dlp for metadata (more reliable than pytube).
    
    Args:
        url: YouTube video URL
        include_video_info: If True, fetches video metadata using yt-dlp
        
    Returns:
        List of Document objects with video transcript and metadata, or error message string.
    """
    try:
        # Get transcript using YoutubeLoader (works reliably)
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,  # We'll get metadata separately with yt-dlp
            language=["en"],
        )
        docs = loader.load()
        
        # Get metadata using yt-dlp if requested
        if include_video_info and docs:
            try:
                video_metadata = _get_video_metadata(url)
                # Merge metadata into the document
                docs[0].metadata.update(video_metadata)
            except Exception as meta_error:
                # If metadata fails, still return transcript with a note
                docs[0].metadata["note"] = f"Metadata unavailable: {str(meta_error)[:50]}"
        
        return docs
        
    except Exception as e:
        error_msg = str(e)
        
        if "400" in error_msg or "Bad Request" in error_msg:
            return (
                "❌ Error: Unable to fetch video details. "
                "This may be due to YouTube API changes or the video being unavailable."
            )
        elif "SSL" in error_msg or "certificate" in error_msg.lower():
            return "❌ SSL Certificate Error: Please check your certificate configuration."
        elif "No transcript found" in error_msg or "transcript" in error_msg.lower():
            return (
                "❌ Error: No transcript available for this video. "
                "The video may not have captions enabled."
            )
        else:
            return f"❌ Error fetching video details: {error_msg}"

def get_youtube_urls_from_playlist(playlist_url: str) -> list[str]:
    """
    Get all video URLs from a YouTube playlist using yt-dlp.
    
    Args:
        playlist_url: YouTube playlist URL
        
    Returns:
        List of video URLs
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,  # Don't download, just extract URLs
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            # Extract video URLs from playlist entries
            video_urls = []
            if 'entries' in info:
                for entry in info['entries']:
                    if entry and 'url' in entry:
                        video_urls.append(entry['url'])
                    elif entry and 'id' in entry:
                        # Construct URL from video ID
                        video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
            return video_urls
        except Exception as e:
            raise ValueError(f"Error extracting playlist URLs: {str(e)}")

def generate_summary(openai_api_key: str, details: list[Document]) -> str:
    """
    Generate a summary of YouTube video transcript using LangChain's built-in map-reduce chain.
    
    Uses load_summarize_chain from langchain-classic which handles:
    1. Splitting documents into chunks
    2. Summarizing each chunk (map step)
    3. Combining all summaries into final summary (reduce step)
    
    Args:
        openai_api_key: OpenAI API key
        details: List of Document objects containing video transcript
        
    Returns:
        Summary string or error message
    """
    if not details or not isinstance(details, list):
        return "❌ Error: No documents provided for summarization."
    
    try:
        # Get transcript content
        transcript = details[0].page_content if details else ""
        
        if not transcript:
            return "❌ Error: No transcript content found."
        
        # Initialize OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0,
        )
        
        # Split transcript into chunks for map-reduce
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,  # Overlap to maintain context between chunks
        )
        chunks = text_splitter.split_documents(details)
        
        # Use LangChain's built-in map-reduce summarization chain
        # chain_type='map_reduce' handles both map and reduce steps automatically
        summarize_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            verbose=False,
        )
        
        # Invoke the chain on the document chunks
        summary = summarize_chain.invoke(chunks)
        
        # The chain returns a dict with 'output_text' key
        return summary.get("output_text", str(summary))
        
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"


def generate_index(openai_api_key: str, details: list[Document]):
    """
    Generate a vector store index from YouTube video documents.
    
    Args:
        openai_api_key: OpenAI API key for embeddings
        details: List of Document objects containing video transcript
        
    Returns:
        InMemoryVectorStore instance
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    documents = text_splitter.split_documents(details)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    index_db = InMemoryVectorStore.from_documents(documents, embeddings)
    return index_db

def create_chain_type_retrieval(openai_api_key: str, index_db: InMemoryVectorStore):
    """
    Create a chain type retrieval from YouTube video documents.
    
    Args:
        openai_api_key: OpenAI API key for embeddings
        index_db: InMemoryVectorStore instance
        
    Returns:
        RetrievalQA instance
    """
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index_db.as_retriever())
    return chain

def generate_answer(chain: RetrievalQA, query: str):
    """
    Generate a response from YouTube video documents.
    
    Args:
        chain: RetrievalQA instance
        query: Query string
        
    Returns:
        Response string
    """
    answer = chain.run(query)
    return answer


def generate_index_playlist(
    openai_api_key: str,
    playlist_url: str
):
    playlist_urls = get_youtube_urls_from_playlist(playlist_url)
    arr_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    for video_url in playlist_urls:
     loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=False,
        language=["en"],
     )
     docs = loader.load()
     arr_documents.extend(docs)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    index_db = InMemoryVectorStore.from_documents(arr_documents, embeddings)
    return index_db