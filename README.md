# videoQA 📹

AI-powered YouTube video analysis and Q&A. Extract transcripts, generate summaries, and ask questions about video content.

## Quick Start

```bash
# Install
uv sync  # or: pip install -e .

# Run
uv run streamlit run app.py
```

1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys)
2. Paste a YouTube video or playlist URL
3. Click "Process" to generate summary
4. Ask questions about the video

## Deploy to Streamlit Cloud

1. Push to GitHub: `git push origin main`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo: `MishalQamar/youtubeGPT`
4. Main file: `app.py`

## Tech Stack

- Streamlit, LangChain, OpenAI GPT-4o-mini
- RAG for intelligent Q&A

## Requirements

- Python 3.10-3.11
- OpenAI API key
- Videos with captions enabled

---

**Built with ❤️ using LangChain, Streamlit, and OpenAI**
