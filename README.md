# videoQA 📹

AI-powered YouTube video analysis and Q&A system. Extract transcripts, generate summaries, and ask questions about video content using RAG (Retrieval Augmented Generation).

## ✨ Features

- Analyze single videos or entire playlists
- AI-generated summaries using OpenAI GPT
- Interactive Q&A about video content
- Rich video metadata display

## 🚀 Quick Start

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/MishalQamar/videoQA.git
cd videoQA
uv sync  # or: pip install -e .
```

### Run Locally

```bash
uv run streamlit run app.py
# or: streamlit run app.py
```

### Usage

1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) in the sidebar
2. Paste a YouTube video or playlist URL
3. Click "Process" to generate summary and enable Q&A
4. Ask questions about the video content

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub: `git push origin main`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository: `MishalQamar/videoQA`
4. Main file: `app.py`
5. (Optional) Add `OPENAI_API_KEY` in Secrets

## 🛠️ Tech Stack

- **Streamlit** - Web interface
- **LangChain** - RAG pipeline & orchestration
- **OpenAI** - GPT-4o-mini & embeddings
- **yt-dlp** - Video metadata extraction

## 📁 Project Structure

```
videoQA/
├── app.py              # Main Streamlit app
├── utils.py            # YouTube processing & RAG functions
├── langchain_*.py      # Tutorial files (LangChain, RAG, LangGraph)
└── requirements.txt    # Dependencies
```

## ⚠️ Requirements

- Python 3.9-3.11
- OpenAI API key
- Videos with captions/transcripts enabled

## 📚 Learning Resources

Includes three comprehensive tutorials:
- `langchain_tutorial.py` - LangChain fundamentals
- `langchain_rag_tutorial.py` - RAG implementation guide
- `langgraph_tutorial.py` - Advanced agent workflows

## 🔧 Troubleshooting

**Virtual environment conflicts:**
```bash
conda deactivate  # or: deactivate
uv sync && uv run streamlit run app.py
```

**Streamlit not found:**
```bash
uv run streamlit run app.py  # Always use uv run
```

## 📝 License

Open source - feel free to use and modify.

---

**Built with ❤️ using LangChain, Streamlit, and OpenAI**
