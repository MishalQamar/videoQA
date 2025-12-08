# videoQA - YouTube Video Question & Answer System

A powerful application that analyzes YouTube videos and allows you to ask questions about their content using AI. Built with Streamlit, LangChain, and OpenAI.

## 🎯 Features

- **📹 Single Video Analysis**: Process individual YouTube videos to extract transcripts, generate summaries, and enable Q&A
- **📚 Playlist Processing**: Analyze entire YouTube playlists with batch processing
- **🤖 AI-Powered Summarization**: Automatically generate concise summaries of video content using OpenAI's GPT models
- **💬 Interactive Q&A**: Ask questions about video content and get accurate answers using Retrieval Augmented Generation (RAG)
- **🔍 Vector Search**: Intelligent semantic search through video transcripts for relevant answers
- **📊 Rich Metadata**: Display video information including title, thumbnail, duration, channel, and more

## 🛠️ Technology Stack

- **Frontend**: Streamlit - Interactive web interface
- **LLM Framework**: LangChain - Orchestration and RAG pipeline
- **AI Models**: OpenAI GPT-4o-mini - Summarization and Q&A
- **Embeddings**: OpenAI Embeddings - Vector representations for semantic search
- **YouTube Processing**: 
  - `yt-dlp` - Video metadata extraction
  - `youtube-transcript-api` - Transcript retrieval
- **Vector Store**: InMemoryVectorStore - For semantic search
- **Text Processing**: RecursiveCharacterTextSplitter - Document chunking

## 📋 Prerequisites

- Python >= 3.13
- OpenAI API key ([Get one here](https://platform.openai.com/account/api-keys))
- YouTube videos with enabled captions/transcripts

## 🚀 Installation

1. **Clone the repository**
   ```bash
   cd videoQA
   ```

2. **Install dependencies using uv** (recommended)
   ```bash
   # Make sure no other virtual environments are active
   # If using conda: conda deactivate
   # If using venv: deactivate
   
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up your OpenAI API key**
   
   The application will prompt you for your API key in the sidebar, or you can set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 💻 Usage

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Or with uv:

```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Application

1. **Enter your OpenAI API key** in the sidebar
2. **Choose a mode**:
   - **Single Video**: Process one YouTube video
   - **Playlist**: Process an entire YouTube playlist
3. **Paste the YouTube URL** (video or playlist)
4. **Click "Process"** to:
   - Fetch video transcript
   - Generate AI summary
   - Create vector index for Q&A
5. **Ask questions** about the video content in the Q&A section

### Example Workflow

1. Enter API key → `sk-xxxxxxx`
2. Select "Single Video"
3. Paste URL → `https://www.youtube.com/watch?v=VIDEO_ID`
4. Click "🚀 Process Video"
5. View summary and video metadata
6. Ask questions like:
   - "What are the main points discussed?"
   - "Who is the speaker?"
   - "What examples were given?"

## 📁 Project Structure

```
videoQA/
├── app.py                    # Main Streamlit application
├── utils.py                  # Utility functions for YouTube processing and RAG
├── main.py                   # Entry point
├── langchain_tutorial.py     # LangChain basics tutorial
├── langchain_rag_tutorial.py # RAG implementation tutorial
├── langgraph_tutorial.py     # LangGraph advanced patterns tutorial
├── pyproject.toml            # Project configuration and dependencies
└── README.md                 # This file
```

## 🔧 Key Components

### `app.py`
The main Streamlit application providing:
- User interface for video/playlist input
- Video metadata display
- Summary generation and display
- Interactive Q&A interface
- Playlist batch processing

### `utils.py`
Core utility functions:

- **`get_youtube_details(url)`**: Fetches video transcript and metadata
- **`get_youtube_urls_from_playlist(url)`**: Extracts all video URLs from a playlist
- **`generate_summary(api_key, details)`**: Creates AI-generated summaries using map-reduce
- **`generate_index(api_key, details)`**: Builds vector store for semantic search
- **`create_chain_type_retrieval(api_key, index_db)`**: Sets up RAG chain for Q&A
- **`generate_answer(chain, query)`**: Processes questions and returns answers

### Tutorial Files

The project includes comprehensive tutorials:

- **`langchain_tutorial.py`**: Learn LangChain basics, prompts, chains, and agents
- **`langchain_rag_tutorial.py`**: Deep dive into Retrieval Augmented Generation
- **`langgraph_tutorial.py`**: Advanced stateful agent workflows with LangGraph

## 🧠 How It Works

1. **Video Processing**:
   - Extracts transcript using `YoutubeLoader`
   - Fetches metadata (title, thumbnail, duration, etc.) using `yt-dlp`

2. **Summarization**:
   - Splits transcript into chunks
   - Uses map-reduce approach with GPT-4o-mini
   - Generates concise summary

3. **Q&A System (RAG)**:
   - Creates vector embeddings of transcript chunks
   - Stores embeddings in InMemoryVectorStore
   - On question:
     - Retrieves relevant transcript chunks via similarity search
     - Passes context + question to LLM
     - Returns grounded answer

## 🎓 Learning Resources

This project includes three comprehensive tutorials:

### LangChain Tutorial (`langchain_tutorial.py`)
Learn the fundamentals:
- Basic chat with OpenAI
- Text summarization
- Document Q&A
- Structured output
- Agents with tools

### RAG Tutorial (`langchain_rag_tutorial.py`)
Master Retrieval Augmented Generation:
- Document loading and splitting
- Embeddings and vector stores
- Retrieval strategies
- RAG chains (LCEL and RetrievalQA)
- Advanced patterns and best practices

### LangGraph Tutorial (`langgraph_tutorial.py`)
Build advanced agent workflows:
- State management
- Nodes and edges
- Conditional routing
- Streaming and checkpointing
- Human-in-the-loop patterns

Run any tutorial file to explore examples and patterns.

## 🔐 Security Notes

- **API Keys**: Never commit your OpenAI API key to version control
- **Environment Variables**: Use environment variables or Streamlit secrets for production
- **SSL Certificates**: The app configures SSL certificate verification for secure connections

## ⚠️ Limitations

- Videos must have captions/transcripts enabled
- Processing large playlists may take time
- Vector store is in-memory (not persistent across sessions)
- Requires active internet connection
- OpenAI API usage incurs costs

## 🔧 Troubleshooting

### Virtual Environment Conflicts

If you see warnings about `VIRTUAL_ENV` not matching the project environment:

```bash
# Deactivate any active conda/venv environments
conda deactivate  # if using conda
deactivate        # if using venv

# Then use uv which manages its own environment
uv sync
uv run streamlit run app.py
```

### Streamlit Command Not Found

If `streamlit` command is not found:

```bash
# Always use uv run to execute commands
uv run streamlit run app.py

# Or activate the project's virtual environment first
source .venv/bin/activate  # On macOS/Linux
streamlit run app.py
```

### SSL Certificate Errors

If you encounter SSL certificate errors:
- The app configures SSL certificates automatically using `certifi`
- Make sure `certifi` is installed: `uv sync`

## 🛣️ Roadmap

Potential future enhancements:
- [ ] Persistent vector store (Chroma, Pinecone)
- [ ] Support for multiple languages
- [ ] Video timestamp references in answers
- [ ] Export summaries and Q&A sessions
- [ ] Multi-video comparison
- [ ] Custom chunking strategies
- [ ] Caching for improved performance

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

## 📞 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Made with ❤️ for learning and exploring AI-powered video content analysis**

