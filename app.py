import ssl
import urllib.request

import certifi
import streamlit as st
from utils import (
    get_youtube_details,
    generate_summary,
    generate_index,
    create_chain_type_retrieval,
    generate_answer,
    get_youtube_urls_from_playlist,
)

# Configure SSL certificate verification
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda *args, **kwargs: ssl_context
urllib.request._opener = None

# Page configuration
st.set_page_config(
    page_title="videoQA - YouTube Video Analyzer",
    page_icon="💂🏻‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "video_details" not in st.session_state:
    st.session_state.video_details = None
if "db_index" not in st.session_state:
    st.session_state.db_index = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = []

# Sidebar
st.sidebar.title("📹 videoQA")
st.sidebar.markdown("---")
user_api_key = st.sidebar.text_input(
    label="**Enter your OpenAI API key**",
    type="password",
    placeholder="sk-xxxxxxx",
    help="You can get your API key from https://platform.openai.com/account/api-keys",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📖 How to use:
    1. Enter your OpenAI API key
    2. Choose Single Video or Playlist
    3. Paste a YouTube URL
    4. Click "Process"
    5. View the summary
    6. Ask questions about the video
    """
)

# Main content
st.title("📹 videoQA")
st.markdown("### YouTube Video Analyzer & Q&A")
st.markdown("---")

if not user_api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
else:
    # Mode selection
    mode = st.radio(
        "**Select Mode:**",
        ["Single Video", "Playlist"],
        horizontal=True,
        help="Choose to process a single video or an entire playlist",
    )
    st.markdown("---")

    if mode == "Single Video":
        # Single video processing
        col1, col2 = st.columns([3, 1])
        with col1:
            youtube_url = st.text_input(
                "**YouTube Video URL:**",
                placeholder="https://www.youtube.com/watch?v=...",
                key="video_input",
                label_visibility="visible",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            process_button = st.button("🚀 Process Video", type="primary", use_container_width=True)

        if youtube_url and process_button:
            with st.spinner("📥 Fetching video details and transcript..."):
                details = get_youtube_details(youtube_url)

            if isinstance(details, str) and details.startswith("❌"):
                st.error(details)
            elif isinstance(details, list) and len(details) > 0:
                st.session_state.video_details = details

                # Display video info
                if details[0].metadata:
                    title = details[0].metadata.get("title", "Unknown Title")
                    thumbnail_url = details[0].metadata.get("thumbnail_url", "")
                    duration = round(details[0].metadata.get("length", 0) / 60, 2)
                    author = details[0].metadata.get("author", "Unknown")

                    st.markdown("## 📹 Video Information")
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if thumbnail_url and thumbnail_url != "Unknown":
                            st.image(thumbnail_url, use_container_width=True)

                    with col2:
                        st.markdown(f"### {title}")
                        st.markdown(f"**👤 Channel:** {author}")
                        st.markdown(f"**⏱️ Duration:** {duration} minutes")
                        st.markdown(f"**📊 Transcript:** ✅ Loaded")

                    st.markdown("---")

                # Generate summary
                with st.spinner("🤖 Generating summary..."):
                    summary = generate_summary(user_api_key, details)
                    if summary.startswith("❌"):
                        st.error(summary)
                    else:
                        st.session_state.summary = summary
                        st.markdown("## 📝 Summary")
                        st.info(summary)
                        st.markdown("---")

                # Generate vector index for Q&A
                with st.spinner("🔍 Indexing video content for Q&A..."):
                    db_index = generate_index(user_api_key, details)
                    st.session_state.db_index = db_index
                    st.success("✅ Video indexed! You can now ask questions.")
                    st.markdown("---")

            else:
                st.warning("⚠️ No content found or unexpected response format.")

    else:
        # Playlist processing
        col1, col2 = st.columns([3, 1])
        with col1:
            playlist_url = st.text_input(
                "**YouTube Playlist URL:**",
                placeholder="https://www.youtube.com/playlist?list=...",
                key="playlist_input",
                label_visibility="visible",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            process_button = st.button("🚀 Process Playlist", type="primary", use_container_width=True)

        if playlist_url and process_button:
            try:
                with st.spinner("📥 Extracting playlist video URLs..."):
                    video_urls = get_youtube_urls_from_playlist(playlist_url)

                if not video_urls:
                    st.warning("⚠️ No videos found in playlist.")
                else:
                    st.success(f"✅ Found {len(video_urls)} videos in playlist")
                    st.markdown("---")

                    # Process each video
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, video_url in enumerate(video_urls, 1):
                        status_text.text(f"Processing video {idx}/{len(video_urls)}...")
                        progress_bar.progress(idx / len(video_urls))

                        with st.expander(f"📹 Video {idx}: {video_url}", expanded=(idx == 1)):
                            try:
                                with st.spinner("Fetching video details..."):
                                    details = get_youtube_details(video_url)

                                if isinstance(details, str) and details.startswith("❌"):
                                    st.error(details)
                                elif isinstance(details, list) and len(details) > 0:
                                    # Display video info
                                    if details[0].metadata:
                                        title = details[0].metadata.get("title", "Unknown Title")
                                        thumbnail_url = details[0].metadata.get("thumbnail_url", "")
                                        duration = round(details[0].metadata.get("length", 0) / 60, 2)

                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            if thumbnail_url and thumbnail_url != "Unknown":
                                                st.image(thumbnail_url, use_container_width=True)
                                        with col2:
                                            st.markdown(f"**{title}**")
                                            st.caption(f"Duration: {duration} minutes")

                                    # Generate summary
                                    with st.spinner("Generating summary..."):
                                        summary = generate_summary(user_api_key, details)
                                        if not summary.startswith("❌"):
                                            st.markdown("**Summary:**")
                                            st.info(summary)

                                    st.session_state.processed_videos.append({
                                        "url": video_url,
                                        "details": details,
                                        "summary": summary,
                                    })
                            except Exception as e:
                                st.error(f"❌ Error processing video: {str(e)}")

                    status_text.text("✅ Playlist processing complete!")
                    progress_bar.empty()
            except Exception as e:
                st.error(f"❌ Error processing playlist: {str(e)}")

    # Q&A Section (show if single video is processed)
    if st.session_state.video_details and st.session_state.db_index:
        st.markdown("---")
        st.markdown("## 💬 Ask Questions About the Video")
        st.markdown("Ask anything about the video content!")

        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "**Your Question:**",
                placeholder="e.g., What are the main points discussed?",
                key="question_input",
                label_visibility="visible",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

        if ask_button and question:
            if not st.session_state.db_index:
                st.error("❌ Video index not available. Please process the video again.")
            else:
                with st.spinner("🤔 Thinking..."):
                    try:
                        chain = create_chain_type_retrieval(user_api_key, st.session_state.db_index)
                        answer = generate_answer(chain, question)

                        st.markdown("### 💡 Answer")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")