# Advanced RAG Assistant 🔍

An intelligent document analysis system built with Streamlit, LangChain, and Ollama that provides enhanced chat, search, summarization, and analytics capabilities with memory-aware interactions.

## Features

✨ **Multi-Modal Document Support**
- PDF files with intelligent text extraction
- CSV files with structured data processing
- Text files (.txt, .md, .markdown) with format preservation
- Batch file processing with metadata tracking

🤖 **Intelligent Chat Interface**
- Context-aware conversations with your documents
- **Chat history integration** - maintains conversation context across interactions
- **Memory-enhanced responses** - references previous conversations when relevant
- Source attribution with clickable references
- Export chat sessions for later review

📊 **Document Analytics**
- Database statistics and comprehensive metrics
- File type distribution analysis
- Document management and filtering tools
- Upload tracking and metadata analysis

🔍 **Enhanced Search**
- **Query processing** - LLM helps interpret and enhance search queries
- Semantic similarity search with relevance scoring
- Configurable result limits and filtering
- Search suggestions and tips

📝 **Smart Summarization**
- **Context-aware summaries** - considers conversation history for better relevance
- All documents or specific document summaries
- Structured summary output with key insights
- Downloadable summary reports

## Architecture

The system uses an **Enhanced RAG** architecture with intelligent query processing:

1. **Document Processing**: Files are loaded, chunked with optimal overlap, and embedded
2. **Vector Storage**: ChromaDB stores document embeddings with rich metadata locally
3. **Query Enhancement**: LLM processes user queries to improve retrieval effectiveness
4. **Memory Integration**: Chat history and context are maintained for coherent conversations
5. **Intelligent Retrieval**: Context-aware search finds relevant document chunks
6. **Response Generation**: Ollama's Mistral model generates responses using retrieved context and conversation memory
7. **Interface**: Streamlit provides an intuitive web interface with persistent sessions

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models (see setup below)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd advanced-rag-assistant
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama models**
```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai/ for installation instructions

# Pull required models
ollama pull mistral          # ~4.1GB - Main chat model
ollama pull nomic-embed-text # ~274MB - Embedding model
```

4. **Create project directories**
```bash
mkdir -p data chroma
```

5. **Run the application**
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
advanced-rag-assistant/
├── main.py                 # Main Streamlit application
├── get_embedding_function.py  # Embedding configuration
├── requirements.txt        # Python dependencies
├── styles.css             # Custom CSS (optional)
├── data/                  # Document storage directory
├── chroma/                # ChromaDB vector database
└── README.md              # This file
```

## Configuration

### Embedding Options

The system supports multiple embedding providers in `get_embedding_function.py`:

**Default: Ollama (Local)**
```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**AWS Bedrock (Cloud)**
```python
embeddings = BedrockEmbeddings(
    credentials_profile_name="default", 
    region_name="us-east-1"
)
```

### Model Configuration

- **Chat Model**: Mistral (via Ollama) - handles query processing and response generation
- **Embedding Model**: nomic-embed-text (via Ollama) - creates document embeddings
- **Vector Store**: ChromaDB (local persistence with metadata)
- **Memory**: In-session chat history with context preservation

## Usage Guide

### 1. Upload Documents
- Use the sidebar file uploader
- Select multiple files (PDF, CSV, TXT, MD)
- Click "Process Files" to add them to the database with metadata

### 2. Chat with Documents
- Ask questions in natural language
- The system maintains conversation context and references previous messages
- **Enhanced query processing**: The LLM may rephrase or expand your query for better results
- View sources for each response with confidence scores
- Export conversation history

### 3. Search Documents  
- Enter keywords or phrases
- **Intelligent search**: System processes your query to improve search effectiveness
- Adjust number of results and view relevance scores
- Context from previous conversations helps refine searches

### 4. Generate Summaries
- Summarize all documents or specific files
- **Memory-aware summaries**: Takes into account your previous questions and interests
- Choose summary scope and download reports
- Progressive summarization improves with more context

### 5. Analyze Document Database
- View database statistics and document relationships
- Filter documents by type, upload date, or content
- Track conversation patterns and frequently accessed topics
- Manage document collection with intelligent insights

## Key Enhanced Features

### Memory-Aware Interactions
- **Conversation Continuity**: References previous messages and maintains context
- **Progressive Learning**: Responses improve as the system learns your preferences
- **Context Integration**: Uses chat history to provide more relevant answers
- **Session Persistence**: Maintains conversation state across application usage

### Intelligent Query Processing
- **Query Enhancement**: LLM analyzes and improves search queries before retrieval
- **Context Addition**: Automatically includes relevant context from conversation history
- **Intent Recognition**: Better understands what you're looking for based on previous interactions
- **Ambiguity Resolution**: Asks clarifying questions when queries are unclear

### Advanced Document Processing
- **Smart Text Splitting**: Optimized chunk sizes for different file types
- **Metadata Enrichment**: Tracks document relationships and usage patterns
- **Encoding Handling**: Multiple encoding support with fallback options
- **Error Recovery**: Robust file processing with detailed error reporting

### User Interface Enhancements
- **Responsive Design**: Modern, professional styling with smooth interactions
- **Real-time Updates**: Dynamic content refresh and progress indicators
- **Export Capabilities**: Download conversations, summaries, and analytics
- **Memory Indicators**: Visual cues showing when context from previous conversations is used

## Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service if needed
ollama serve
```

**2. Model Not Found**
```bash
# Pull missing models
ollama pull mistral
ollama pull nomic-embed-text
```

**3. Memory/Context Issues**
- Clear chat history if responses seem inconsistent
- Restart application to reset conversation context
- Check if ChromaDB directory has proper permissions

**4. File Processing Errors**
- Verify file format compatibility
- Check file encoding for text files
- Ensure sufficient disk space for document processing

### Performance Optimization

**For Large Document Collections:**
- Regular database maintenance and cleanup
- Use specific document filters to narrow search scope
- Monitor memory usage during batch processing

**For Better Conversational Experience:**
- Start with clear, specific questions
- Build on previous questions for deeper analysis
- Use document names when referencing specific files
- Export important conversations for future reference

## Development

### Adding New Document Types

1. Update `SUPPORTED_EXTENSIONS` in `main.py`
2. Create loader function (e.g., `load_single_docx`)
3. Add processing logic in `process_uploaded_file`
4. Update metadata handling for new file type

### Enhancing Memory Features

```python
# Extend conversation memory
def enhance_context_integration(messages, current_query):
    relevant_history = filter_relevant_messages(messages, current_query)
    enhanced_context = build_enhanced_context(relevant_history)
    return enhanced_context
```

### Custom Query Processing

```python
# Customize query enhancement
def enhance_user_query(query, conversation_context):
    enhanced_query = llm.process_with_context(query, conversation_context)
    return enhanced_query
```

## API Integration

The system integrates with:
- **Ollama**: Local model inference for chat and query processing
- **AWS Bedrock**: Cloud-based embeddings (optional)
- **ChromaDB**: Vector database with metadata storage
- **LangChain**: Framework for enhanced RAG capabilities

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama and LangChain documentation
3. Open an issue on the repository

## Acknowledgments

- **LangChain**: Enhanced RAG framework and memory integration
- **Streamlit**: Interactive web interface framework
- **Ollama**: Local LLM inference for query processing and chat
- **ChromaDB**: Vector database with metadata support
- **Mistral**: Language model for intelligent responses

---

**Built by Ben Slama Latifa, Final-year Data Science Engineering Student**