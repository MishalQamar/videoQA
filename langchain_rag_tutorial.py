"""
============================================================================
LANGCHAIN RAG (RETRIEVAL AUGMENTED GENERATION) TUTORIAL
============================================================================

A comprehensive guide to building RAG applications with LangChain.

RAG combines:
1. Retrieval: Finding relevant information from a knowledge base
2. Augmentation: Adding that information to the LLM's context
3. Generation: Using the LLM to generate answers based on the retrieved context

This tutorial covers everything from basic RAG to advanced patterns.
"""

# ============================================================================
# PART 1: ESSENTIAL IMPORTS FOR RAG
# ============================================================================

# Core LangChain components
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Document loaders (load data from various sources)
from langchain_community.document_loaders import (
    YoutubeLoader,
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
)

# Text splitters (chunk documents for better retrieval)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (convert text to vectors)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Prompts (structure instructions for the LLM)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Legacy chains (for simple RAG patterns)
from langchain_classic.chains import RetrievalQA

# Utilities
import os
from typing import List, Dict, Any


# ============================================================================
# PART 2: UNDERSTANDING RAG - THE BIG PICTURE
# ============================================================================

"""
RAG WORKFLOW:

1. LOAD: Load documents from various sources (PDFs, web pages, YouTube, etc.)
   └─> Result: List of Document objects

2. SPLIT: Split large documents into smaller chunks
   └─> Result: List of smaller Document chunks

3. EMBED: Convert text chunks into vector embeddings
   └─> Result: Vector representations of text

4. STORE: Store embeddings in a vector database
   └─> Result: Searchable vector store

5. RETRIEVE: Given a query, find relevant chunks using similarity search
   └─> Result: Top-K most relevant document chunks

6. AUGMENT: Add retrieved chunks to the LLM's prompt as context
   └─> Result: Prompt with query + context

7. GENERATE: LLM generates answer based on query + context
   └─> Result: Final answer

WHY RAG?
- LLMs have limited context windows
- LLMs can't access real-time or private data
- RAG allows LLMs to answer questions about YOUR data
- Reduces hallucinations by grounding answers in retrieved documents
"""


# ============================================================================
# PART 3: STEP 1 - LOADING DOCUMENTS
# ============================================================================

def example_1_load_documents():
    """
    Load documents from various sources.
    Documents are the foundation of RAG - they contain the knowledge base.
    """
    print("\n--- Example 1: Loading Documents ---")
    
    # Option 1: Load from YouTube
    # loader = YoutubeLoader.from_youtube_url(
    #     "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    #     add_video_info=False,
    #     language=["en"]
    # )
    # docs = loader.load()
    
    # Option 2: Load from web pages
    # loader = WebBaseLoader("https://example.com/article")
    # docs = loader.load()
    
    # Option 3: Load from PDF
    # loader = PyPDFLoader("path/to/document.pdf")
    # docs = loader.load()
    
    # Option 4: Load from text file
    # loader = TextLoader("path/to/file.txt")
    # docs = loader.load()
    
    # Option 5: Create documents manually
    docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"source": "intro.txt", "page": 1}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"source": "ml.txt", "page": 1}
        ),
        Document(
            page_content="LangChain is a framework for building applications with LLMs, providing tools for RAG, agents, and more.",
            metadata={"source": "langchain.txt", "page": 1}
        ),
    ]
    
    print(f"Loaded {len(docs)} documents")
    print(f"First document: {docs[0].page_content[:50]}...")
    return docs


# ============================================================================
# PART 4: STEP 2 - SPLITTING DOCUMENTS
# ============================================================================

def example_2_split_documents(docs: List[Document]):
    """
    Split documents into smaller chunks.
    
    Why split?
    - LLMs have token limits
    - Smaller chunks improve retrieval precision
    - Allows finding specific information within large documents
    
    Key parameters:
    - chunk_size: Maximum characters per chunk
    - chunk_overlap: Characters to overlap between chunks (maintains context)
    """
    print("\n--- Example 2: Splitting Documents ---")
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Maximum characters per chunk
        chunk_overlap=50,      # Overlap between chunks (maintains context)
        length_function=len,    # Function to measure chunk size
        add_start_index=True,  # Track where each chunk starts in original doc
    )
    
    # Split documents
    chunks = text_splitter.split_documents(docs)
    
    print(f"Original documents: {len(docs)}")
    print(f"After splitting: {len(chunks)} chunks")
    print(f"First chunk: {chunks[0].page_content[:100]}...")
    print(f"Chunk metadata: {chunks[0].metadata}")
    
    return chunks


# ============================================================================
# PART 5: STEP 3 - CREATING EMBEDDINGS
# ============================================================================

def example_3_create_embeddings(api_key: str):
    """
    Create embeddings to convert text into vectors.
    
    Embeddings are numerical representations of text that capture semantic meaning.
    Similar texts have similar embeddings, enabling semantic search.
    """
    print("\n--- Example 3: Creating Embeddings ---")
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # OpenAI embedding model
        api_key=api_key,
    )
    
    # Embed a single text
    text = "Python is a programming language"
    embedding = embeddings.embed_query(text)
    
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    return embeddings


# ============================================================================
# PART 6: STEP 4 - CREATING A VECTOR STORE
# ============================================================================

def example_4_create_vector_store(chunks: List[Document], embeddings: OpenAIEmbeddings):
    """
    Create a vector store to store and search document embeddings.
    
    Vector stores enable:
    - Fast similarity search
    - Retrieval of relevant documents based on query
    - Scalable storage of embeddings
    """
    print("\n--- Example 4: Creating Vector Store ---")
    
    # Create vector store from documents
    # This automatically:
    # 1. Generates embeddings for all chunks
    # 2. Stores them in the vector store
    # 3. Makes them searchable
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    print(f"Vector store created with {len(chunks)} documents")
    print("Vector store is ready for similarity search!")
    
    return vectorstore


# ============================================================================
# PART 7: STEP 5 - RETRIEVAL (SIMILARITY SEARCH)
# ============================================================================

def example_5_retrieval(vectorstore: InMemoryVectorStore, query: str):
    """
    Retrieve relevant documents using similarity search.
    
    Similarity search finds documents whose embeddings are closest to the query embedding.
    This is the "Retrieval" part of RAG.
    """
    print("\n--- Example 5: Retrieval (Similarity Search) ---")
    
    # Perform similarity search
    # k=3 means retrieve top 3 most similar documents
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n  {i}. {doc.page_content[:100]}...")
        print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
    
    return retrieved_docs


# ============================================================================
# PART 8: STEP 6 - CREATING A RETRIEVER
# ============================================================================

def example_6_create_retriever(vectorstore: InMemoryVectorStore):
    """
    Create a retriever from a vector store.
    
    Retrievers are Runnable objects that can be chained with LLMs.
    They provide a standardized interface for retrieval.
    """
    print("\n--- Example 6: Creating Retriever ---")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use similarity search
        search_kwargs={"k": 3},    # Retrieve top 3 documents
    )
    
    # Test the retriever
    query = "What is Python?"
    results = retriever.invoke(query)
    
    print(f"Retriever created successfully")
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents")
    
    return retriever


# ============================================================================
# PART 9: STEP 7 - SIMPLE RAG CHAIN (LCEL - LangChain Expression Language)
# ============================================================================

def example_7_simple_rag_chain(api_key: str, retriever):
    """
    Build a simple RAG chain using LCEL.
    
    This is the modern LangChain way to build RAG:
    retriever -> format_docs -> prompt -> llm -> output_parser
    """
    print("\n--- Example 7: Simple RAG Chain (LCEL) ---")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer questions. "
                  "If you don't know the answer based on the context, say so."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # Format function: Combine retrieved documents into a single string
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the RAG chain using LCEL
    # The pipe operator (|) chains components together
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Use the chain
    query = "What is Python?"
    answer = rag_chain.invoke(query)
    
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    
    return rag_chain


# ============================================================================
# PART 10: STEP 8 - RAG WITH RETRIEVALQA (LEGACY APPROACH)
# ============================================================================

def example_8_retrieval_qa(api_key: str, vectorstore: InMemoryVectorStore):
    """
    Build RAG using RetrievalQA chain (legacy but simpler approach).
    
    RetrievalQA handles the entire RAG pipeline automatically.
    Good for quick prototyping, but less flexible than LCEL.
    """
    print("\n--- Example 8: RAG with RetrievalQA ---")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create RetrievalQA chain
    # chain_type="stuff" means put all retrieved docs into the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Options: "stuff", "map_reduce", "refine", "map_rerank"
        retriever=retriever,
        return_source_documents=True,  # Return source docs for citation
    )
    
    # Use the chain
    query = "What is machine learning?"
    result = qa_chain.invoke({"query": query})
    
    print(f"Question: {query}")
    print(f"Answer: {result['result']}")
    print(f"Source documents: {len(result.get('source_documents', []))}")
    
    return qa_chain


# ============================================================================
# PART 11: ADVANCED RAG PATTERNS
# ============================================================================

def example_9_advanced_rag_patterns(api_key: str, vectorstore: InMemoryVectorStore):
    """
    Advanced RAG patterns for better performance.
    """
    print("\n--- Example 9: Advanced RAG Patterns ---")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Pattern 1: Multi-query retrieval
    # Generate multiple queries from the original query to improve retrieval
    print("\n1. Multi-Query Retrieval:")
    print("   Generate multiple query variations to retrieve more relevant docs")
    
    # Pattern 2: Re-ranking
    # Re-rank retrieved documents to get the most relevant ones
    print("\n2. Re-ranking:")
    print("   Use a cross-encoder to re-rank retrieved documents")
    
    # Pattern 3: Contextual compression
    # Compress retrieved documents to only relevant parts
    print("\n3. Contextual Compression:")
    print("   Extract only relevant sentences from retrieved documents")
    
    # Pattern 4: Parent document retriever
    # Retrieve parent documents when child chunks are found
    print("\n4. Parent Document Retriever:")
    print("   Retrieve full parent documents when child chunks match")
    
    # Pattern 5: Self-query retriever
    # Extract metadata filters from the query
    print("\n5. Self-Query Retriever:")
    print("   Extract metadata filters (e.g., date, author) from natural language queries")


# ============================================================================
# PART 12: RAG WITH SOURCE CITATIONS
# ============================================================================

def example_10_rag_with_citations(api_key: str, retriever):
    """
    RAG that includes source citations in the answer.
    """
    print("\n--- Example 10: RAG with Source Citations ---")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    # Enhanced prompt that requests citations
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions using the provided context. "
                  "Always cite your sources using [Source: filename] format."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    def format_docs_with_sources(docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[{i}] {doc.page_content}\n   Source: {source}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    query = "What is Python?"
    answer = rag_chain.invoke(query)
    
    print(f"Question: {query}")
    print(f"Answer with citations:\n{answer}")


# ============================================================================
# PART 13: COMPLETE RAG PIPELINE EXAMPLE
# ============================================================================

def example_11_complete_rag_pipeline(api_key: str):
    """
    Complete end-to-end RAG pipeline example.
    """
    print("\n--- Example 11: Complete RAG Pipeline ---")
    
    # Step 1: Load documents
    docs = [
        Document(
            page_content="Python is a high-level, interpreted programming language. It emphasizes code readability and simplicity.",
            metadata={"source": "python_guide.txt"}
        ),
        Document(
            page_content="Machine learning algorithms learn patterns from data to make predictions or decisions.",
            metadata={"source": "ml_basics.txt"}
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications including RAG, agents, and chains.",
            metadata={"source": "langchain_docs.txt"}
        ),
    ]
    
    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    
    # Step 3: Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Step 4: Create vector store
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    
    # Step 5: Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Step 6: Create RAG chain
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the provided context. "
                  "If the answer is not in the context, say 'I don't have enough information.'"),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(f"- {doc.page_content}" for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Step 7: Query the RAG system
    questions = [
        "What is Python?",
        "How does machine learning work?",
        "What is LangChain?",
    ]
    
    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    return rag_chain


# ============================================================================
# PART 14: RAG BEST PRACTICES
# ============================================================================

"""
RAG BEST PRACTICES:

1. CHUNK SIZE:
   - Too small: Loses context, too many chunks
   - Too large: May exceed token limits, less precise retrieval
   - Sweet spot: 500-1000 characters for most use cases
   - Use overlap (50-200 chars) to maintain context between chunks

2. EMBEDDING MODELS:
   - Use models optimized for retrieval (e.g., text-embedding-3-small)
   - Consider domain-specific embeddings for specialized knowledge
   - Test different models to find the best for your use case

3. RETRIEVAL:
   - Start with k=3-5 documents
   - Use similarity search for semantic queries
   - Consider hybrid search (semantic + keyword) for better results
   - Re-rank results if needed for better precision

4. PROMPT ENGINEERING:
   - Clearly instruct the LLM to use only provided context
   - Ask for citations when needed
   - Handle cases where context doesn't contain the answer
   - Use few-shot examples for complex tasks

5. EVALUATION:
   - Test with diverse queries
   - Measure retrieval accuracy (are relevant docs retrieved?)
   - Measure generation quality (are answers correct and grounded?)
   - Use metrics like precision, recall, and answer accuracy

6. OPTIMIZATION:
   - Cache embeddings to avoid recomputation
   - Use async operations for better performance
   - Consider using a persistent vector store (e.g., Chroma, Pinecone) for production
   - Monitor token usage and costs

7. ERROR HANDLING:
   - Handle cases where no relevant documents are found
   - Handle API failures gracefully
   - Validate retrieved documents before passing to LLM
   - Provide fallback responses

8. SECURITY:
   - Don't expose API keys
   - Validate and sanitize user queries
   - Be careful with sensitive data in documents
   - Consider data privacy and compliance requirements
"""


# ============================================================================
# PART 15: COMMON RAG PATTERNS CHEAT SHEET
# ============================================================================

"""
RAG PATTERNS CHEAT SHEET:

1. SIMPLE RAG (Stuff):
   - Put all retrieved docs into prompt
   - Best for: Small number of documents, simple queries
   - Pros: Simple, fast
   - Cons: Limited by token limits

2. MAP-REDUCE:
   - Summarize each doc separately, then combine summaries
   - Best for: Large number of documents
   - Pros: Handles many documents
   - Cons: More LLM calls, higher cost

3. REFINE:
   - Iteratively refine answer with each document
   - Best for: When you want to build up answer incrementally
   - Pros: Can incorporate all documents
   - Cons: Sequential processing, slower

4. MAP-RERANK:
   - Score each document's relevance, return best
   - Best for: When you need the single best answer
   - Pros: High precision
   - Cons: More complex, requires scoring model

5. AGENTIC RAG:
   - LLM decides when to retrieve and what to retrieve
   - Best for: Complex, multi-step queries
   - Pros: Flexible, can handle follow-ups
   - Cons: More complex, slower

6. SELF-RAG:
   - LLM evaluates its own retrieval and generation
   - Best for: High-quality, self-correcting systems
   - Pros: Self-improving
   - Cons: Very complex, expensive
"""


# ============================================================================
# PART 16: TROUBLESHOOTING COMMON RAG ISSUES
# ============================================================================

"""
COMMON RAG ISSUES AND SOLUTIONS:

1. ISSUE: Irrelevant documents retrieved
   SOLUTION:
   - Improve chunking strategy
   - Use better embedding models
   - Increase chunk overlap
   - Try hybrid search (semantic + keyword)
   - Re-rank retrieved documents

2. ISSUE: Answer not grounded in context
   SOLUTION:
   - Strengthen prompt to emphasize using only context
   - Add examples in prompt
   - Use structured output to force citations
   - Post-process to validate answer is in context

3. ISSUE: Missing information in retrieved docs
   SOLUTION:
   - Increase k (retrieve more documents)
   - Improve chunking (smaller chunks, better overlap)
   - Use multi-query retrieval
   - Consider expanding query with synonyms

4. ISSUE: Token limit exceeded
   SOLUTION:
   - Reduce chunk size
   - Reduce k (retrieve fewer documents)
   - Use map-reduce instead of stuff
   - Compress retrieved documents

5. ISSUE: Slow retrieval
   SOLUTION:
   - Use async operations
   - Cache embeddings
   - Use faster vector stores
   - Optimize chunk size

6. ISSUE: High costs
   SOLUTION:
   - Use cheaper embedding models
   - Reduce number of retrieved documents
   - Cache embeddings and results
   - Use smaller LLM models
"""


# ============================================================================
# PART 17: PRODUCTION RAG CONSIDERATIONS
# ============================================================================

"""
PRODUCTION RAG CHECKLIST:

1. VECTOR STORE:
   - [ ] Use persistent vector store (Chroma, Pinecone, Weaviate, etc.)
   - [ ] Set up proper indexing
   - [ ] Plan for scalability
   - [ ] Monitor storage costs

2. EMBEDDINGS:
   - [ ] Cache embeddings to avoid recomputation
   - [ ] Use batch processing for large datasets
   - [ ] Monitor embedding API costs
   - [ ] Consider fine-tuned embeddings for domain-specific data

3. RETRIEVAL:
   - [ ] Tune retrieval parameters (k, similarity threshold)
   - [ ] Implement hybrid search if needed
   - [ ] Add query preprocessing (spell check, expansion)
   - [ ] Monitor retrieval quality

4. GENERATION:
   - [ ] Set appropriate temperature for consistency
   - [ ] Implement streaming for better UX
   - [ ] Add response validation
   - [ ] Monitor generation quality and costs

5. MONITORING:
   - [ ] Track retrieval accuracy
   - [ ] Track answer quality
   - [ ] Monitor latency
   - [ ] Track costs
   - [ ] Set up alerts for failures

6. SECURITY:
   - [ ] Secure API keys
   - [ ] Validate user inputs
   - [ ] Implement rate limiting
   - [ ] Audit data access
   - [ ] Comply with data privacy regulations

7. TESTING:
   - [ ] Unit tests for each component
   - [ ] Integration tests for full pipeline
   - [ ] Evaluation on test dataset
   - [ ] A/B testing for different configurations
"""


# ============================================================================
# 🧪 TEST YOUR KNOWLEDGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 LANGCHAIN RAG TUTORIAL")
    print("="*70)
    print("\nThis tutorial covers:")
    print("  ✓ Document loading and splitting")
    print("  ✓ Embeddings and vector stores")
    print("  ✓ Retrieval and RAG chains")
    print("  ✓ Advanced patterns and best practices")
    print("\nUncomment examples below to try them!")
    print("(Make sure to set your OPENAI_API_KEY first)\n")
    
    # Example usage (uncomment to run):
    # api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
    # 
    # # Run examples
    # docs = example_1_load_documents()
    # chunks = example_2_split_documents(docs)
    # embeddings = example_3_create_embeddings(api_key)
    # vectorstore = example_4_create_vector_store(chunks, embeddings)
    # example_5_retrieval(vectorstore, "What is Python?")
    # retriever = example_6_create_retriever(vectorstore)
    # rag_chain = example_7_simple_rag_chain(api_key, retriever)
    # example_8_retrieval_qa(api_key, vectorstore)
    # example_11_complete_rag_pipeline(api_key)
    
    print("\n✅ RAG Tutorial complete! You now know how to build RAG applications!")
    print("\nNext steps:")
    print("  1. Try building a RAG system for your own documents")
    print("  2. Experiment with different chunk sizes and retrieval strategies")
    print("  3. Evaluate your RAG system's performance")
    print("  4. Deploy to production with proper monitoring")

