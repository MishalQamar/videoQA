# videoQA 📹

AI-powered YouTube video analysis and Q&A. Extract transcripts, generate summaries, and ask questions about video content.

## What It Does

- Analyzes YouTube videos or playlists
- Generates AI summaries of video content
- Answers questions about videos using AI
- Displays video metadata and information

## Quick Start

```bash
# Install
uv sync

# Run
uv run streamlit run app.py
```

## Usage

1. Enter your OpenAI API key
2. Paste a YouTube video or playlist URL
3. Click "Process" to generate summary
4. Ask questions about the video

## Deploy

Push to GitHub and deploy on Streamlit Cloud:
- Repository: MishalQamar/youtubeGPT
- Main file: app.py

## Requirements

- Python 3.10-3.11
- OpenAI API key
- Videos with captions enabled

---

Built with Streamlit, LangChain, and OpenAI
