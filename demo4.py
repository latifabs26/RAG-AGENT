import streamlit as st
import sys
import io
import os
import tempfile
import shutil
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
import time
import pandas as pd
from collections import Counter
import re
from datetime import datetime

# Set UTF-8 encoding for the environment
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv',
    '.md': 'text',
    '.markdown': 'text',
    '.txt': 'text',
}

# Enhanced prompt templates
CHAT_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context and chat history.

Chat History:
{chat_history}

Context from documents:
{context}

---

Current Question: {question}

Please provide a detailed answer based on the context above. If the chat history is relevant to the current question, consider it in your response. If you cannot find the answer in the provided context, say "I don't have enough information in the provided documents to answer this question."
"""

SUMMARY_PROMPT_TEMPLATE = """
You are an expert summarizer. Based on the following documents, create a comprehensive summary.

Documents:
{context}

Please provide a well-structured summary that covers:
1. Main topics and themes
2. Key findings or insights
3. Important details and statistics
4. Overall conclusions

Summary:
"""

def load_css():
    """Load CSS from external file"""
    css_file = Path("styles.css")
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback to basic professional styling
        st.markdown("""
        <style>
        .main { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1e293b;
        }
        .custom-header { 
            background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
            padding: 3rem 2rem; 
            border-radius: 1rem; 
            text-align: center; 
            margin-bottom: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .custom-header h1 { 
            color: white; 
            font-size: 2.5rem; 
            font-weight: 700;
            margin: 0; 
            letter-spacing: -0.025em;
        }
        .custom-header p { 
            color: rgba(255,255,255,0.9); 
            font-size: 1.125rem; 
            margin: 0.5rem 0 0 0;
        }
        </style>
        """, unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system components"""
    try:
        embedding_function = get_embedding_function()
        
        # Ensure chroma directory exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        model = Ollama(model="mistral")
        return db, model
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        raise e

def load_single_pdf(file_path):
    """Load a single PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load PDF {Path(file_path).name}: {e}")
        return []

def load_single_csv(file_path):
    """Load a single CSV file with error handling"""
    try:
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e1:
        try:
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={'delimiter': ','}
            )
            return loader.load()
        except Exception as e2:
            try:
                loader = CSVLoader(
                    file_path=file_path,
                    encoding='utf-8',
                    csv_args={'delimiter': ';'}
                )
                return loader.load()
            except Exception as e3:
                st.error(f"Failed to load CSV {Path(file_path).name}: Multiple encoding/delimiter attempts failed")
                return []

def load_single_text(file_path):
    """Load a single text file with encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            continue
    
    st.error(f"Failed to load text file {Path(file_path).name} with any encoding")
    return []

def process_uploaded_file(uploaded_file, temp_dir):
    """Process a single uploaded file and return documents"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return []
    
    # Save uploaded file to temp directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load documents based on file type
    file_type = SUPPORTED_EXTENSIONS[file_extension]
    
    if file_type == 'pdf':
        documents = load_single_pdf(temp_file_path)
    elif file_type == 'csv':
        documents = load_single_csv(temp_file_path)
    elif file_type == 'text':
        documents = load_single_text(temp_file_path)
    else:
        return []
    
    # Update metadata to include original filename
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["uploaded"] = True
        doc.metadata["upload_time"] = datetime.now().isoformat()
    
    return documents

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        
        try:
            if source.lower().endswith('.csv'):
                chunks = csv_splitter.split_documents([doc])
            else:
                chunks = text_splitter.split_documents([doc])
            
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error splitting document {source}: {e}")
    
    return all_chunks

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks"""
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        if source.lower().endswith('.pdf'):
            page = chunk.metadata.get("page", 0)
            current_source_id = f"{source}:{page}"
        elif source.lower().endswith('.csv'):
            row = chunk.metadata.get("row", 0)
            current_source_id = f"{source}:{row}"
        else:
            current_source_id = f"{source}:0"

        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id
        chunk.metadata["id"] = chunk_id

    return chunks

def add_documents_to_chroma(chunks: list[Document], db):
    """Add new documents to the Chroma database"""
    if not chunks:
        return 0
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        return len(new_chunks)
    else:
        return 0

def format_chat_history(messages):
    """Format chat history for the prompt"""
    if not messages:
        return "No previous conversation."
    
    formatted_history = []
    for message in messages[-6:]:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"Assistant: {message.content}")
    
    return "\n".join(formatted_history)

def query_rag_with_history(query_text: str, chat_history, db, model):
    """Query RAG system with chat history context"""
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    history_text = format_chat_history(chat_history)
    
    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, 
        question=query_text,
        chat_history=history_text
    )
    
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    
    return response_text, sources

def generate_summary(db, model, doc_filter=None):
    """Generate a summary of documents in the database"""
    try:
        # Get all documents or filtered documents
        if doc_filter:
            # Filter documents by source
            all_items = db.get(include=["documents", "metadatas"])
            filtered_docs = []
            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata and metadata.get("source") == doc_filter:
                    filtered_docs.append(all_items["documents"][i])
            context_text = "\n\n---\n\n".join(filtered_docs[:10])  # Limit to 10 chunks
        else:
            # Get top documents by similarity to "summary overview main points"
            results = db.similarity_search("summary overview main points", k=10)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        if not context_text.strip():
            return "No documents available for summarization."
        
        prompt_template = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text)
        
        summary = model.invoke(prompt)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def search_documents(db, query, k=10):
    """Search documents and return results with scores"""
    try:
        results = db.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def get_database_stats(db):
    """Get comprehensive statistics about the database"""
    try:
        existing_items = db.get(include=["metadatas"])
        total_docs = len(existing_items["ids"])
        
        sources = set()
        file_types = Counter()
        upload_dates = []
        
        for metadata in existing_items["metadatas"]:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
                file_ext = Path(metadata["source"]).suffix.lower()
                file_types[file_ext] += 1
                
                if "upload_time" in metadata:
                    upload_dates.append(metadata["upload_time"])
        
        return {
            "total_chunks": total_docs,
            "unique_sources": len(sources),
            "sources": list(sources),
            "file_types": dict(file_types),
            "upload_dates": upload_dates
        }
    except:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "sources": [],
            "file_types": {},
            "upload_dates": []
        }

def create_custom_header():
    """Create a sophisticated custom header"""
    st.markdown("""
    <div class="custom-header fade-in">
        <h1>Advanced RAG Assistant</h1>
        <p>Intelligent Document Analysis • Chat • Search • Summarize</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon_class="metric"):
    """Create a sophisticated metric card"""
    st.markdown(f"""
    <div class="metric-card slide-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources_beautifully(sources):
    """Display sources in a clean format"""
    if sources:
        st.markdown("**Sources:**")
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item slide-in">
                <strong>{i}.</strong> {source}
            </div>
            """, unsafe_allow_html=True)

def display_search_results(results):
    """Display search results in a clean format"""
    if not results:
        st.info("No results found.")
        return
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        
        st.markdown(f"""
        <div class="search-result slide-in">
            <div class="search-result-score">{score:.3f}</div>
            <div class="search-result-title">{source}</div>
            <div class="search-result-content">{content_preview}</div>
        </div>
        """, unsafe_allow_html=True)

def chat_interface():
    """Clean chat interface implementation"""
    st.markdown("### 💬 Chat with Your Documents")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                st.markdown(f"""
                <div class="user-message slide-in">
                    <strong>You:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f"""
                <div class="assistant-message slide-in">
                    <strong>🤖 Assistant:</strong><br>{message.content}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("Ask me anything about your documents...")
    
    if user_question:
        # Add user message immediately
        st.session_state.messages.append(HumanMessage(user_question))
        
        # Create response
        with st.spinner("Analyzing..."):
            try:
                response, sources = query_rag_with_history(
                    user_question, 
                    st.session_state.messages[:-1],
                    st.session_state.db, 
                    st.session_state.model
                )
                
                # Add AI response
                st.session_state.messages.append(AIMessage(response))
                
                # Store sources for display
                st.session_state.last_sources = sources
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(error_msg))
                st.rerun()
    
    # Display sources from last query if available
    if (hasattr(st.session_state, 'last_sources') and 
        st.session_state.last_sources and 
        not user_question):
        with st.expander("View Sources from Last Query", expanded=False):
            display_sources_beautifully(st.session_state.last_sources)

def summarizer_interface():
    """Clean summarizer interface implementation"""
    st.markdown("### Document Summarizer")
    
    stats = get_database_stats(st.session_state.db)
    
    if stats["total_chunks"] == 0:
        st.info("Upload documents first to generate summaries.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_type = st.selectbox(
            "Summary Type",
            ["All Documents", "Specific Document"],
            help="Choose whether to summarize all documents or a specific one",
            key="summary_type_selector"
        )
        
        selected_doc = None
        if summary_type == "Specific Document" and stats["sources"]:
            selected_doc = st.selectbox(
                "Select Document",
                stats["sources"],
                help="Choose a specific document to summarize",
                key="doc_selector"
            )
    
    with col2:
        if st.button("Generate Summary", type="primary", use_container_width=True, key="generate_summary_btn"):
            with st.spinner("Generating summary..."):
                if summary_type == "All Documents":
                    summary = generate_summary(st.session_state.db, st.session_state.model)
                else:
                    summary = generate_summary(st.session_state.db, st.session_state.model, selected_doc)
                
                # Store summary in session state
                st.session_state.current_summary = summary
                st.session_state.summary_type_used = summary_type
                st.session_state.selected_doc_used = selected_doc
    
    # Display stored summary
    if hasattr(st.session_state, 'current_summary'):
        st.markdown(f"""
        <div class="summary-card fade-in">
            <div class="summary-title">
                Summary - {st.session_state.summary_type_used}
                {f": {st.session_state.selected_doc_used}" if st.session_state.selected_doc_used else ""}
            </div>
            <div class="summary-content">{st.session_state.current_summary}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Download summary
        st.download_button(
            "Download Summary",
            st.session_state.current_summary,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_summary_btn"
        )

def search_interface():
    """Clean search interface implementation"""
    st.markdown("### Document Search")
    
    stats = get_database_stats(st.session_state.db)
    
    if stats["total_chunks"] == 0:
        st.info("Upload documents first to enable search functionality.")
        return
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter keywords to search for...",
            help="Search across all your uploaded documents",
            key="search_input"
        )
    
    with col2:
        num_results = st.selectbox("Results", [5, 10, 15, 20], index=1, key="results_selector")
    
    with col3:
        search_button = st.button("Search", type="primary", use_container_width=True, key="search_btn")
    
    if search_query and search_button:
        with st.spinner("Searching documents..."):
            results = search_documents(st.session_state.db, search_query, k=num_results)
            # Store results in session state
            st.session_state.search_results = results
            st.session_state.search_query_used = search_query
    
    # Display stored search results
    if hasattr(st.session_state, 'search_results'):
        if st.session_state.search_results:
            st.markdown(f"**Found {len(st.session_state.search_results)} results for:** `{st.session_state.search_query_used}`")
            display_search_results(st.session_state.search_results)
        else:
            st.info("No results found. Try different keywords.")
    
    # Search suggestions
    st.markdown("**Search Tips:**")
    st.markdown("""
    - Use specific keywords related to your documents
    - Try different phrasings if you don't get results
    - Use quotes for exact phrases: "machine learning"
    - Combine multiple keywords for better results
    """)

def analytics_interface():
    """Clean analytics and filter interface"""
    st.markdown("### Document Analytics & Management")
    
    stats = get_database_stats(st.session_state.db)
    
    if stats["total_chunks"] == 0:
        st.info("Upload documents first to view analytics.")
        return
    
    # Statistics Overview
    st.markdown("#### Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Chunks", stats["total_chunks"])
    with col2:
        create_metric_card("Documents", stats["unique_sources"])
    with col3:
        file_type_count = len(stats["file_types"])
        create_metric_card("File Types", file_type_count)
    with col4:
        create_metric_card("Upload Sessions", len(stats["upload_dates"]))
    
    # File Type Distribution
    if stats["file_types"]:
        st.markdown("#### File Type Distribution")
        file_types_df = pd.DataFrame(
            list(stats["file_types"].items()), 
            columns=["File Type", "Count"]
        )
        st.bar_chart(file_types_df.set_index("File Type"))
    
    # Document Management
    st.markdown("#### Document Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if stats["sources"]:
            filter_type = st.selectbox(
                "Filter by",
                ["All Documents", "PDF Files", "CSV Files", "Text Files"],
                help="Filter documents by file type",
                key="filter_selector"
            )
    
    with col2:
        # Database clearing with confirmation
        if st.button("Clear Database", type="secondary", use_container_width=True, key="clear_db_btn"):
            st.session_state.show_confirm_delete = True
    
    # Show confirmation dialog
    if getattr(st.session_state, 'show_confirm_delete', False):
        st.warning("This will permanently delete all documents from the database!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Delete", type="primary", key="confirm_delete_btn"):
                try:
                    st.session_state.db.delete_collection()
                    st.session_state.db, st.session_state.model = initialize_rag_system()
                    st.session_state.messages = []
                    st.session_state.show_confirm_delete = False
                    st.success("Database cleared successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")
        
        with col2:
            if st.button("Cancel", key="cancel_delete_btn"):
                st.session_state.show_confirm_delete = False
                st.rerun()
    
    # Filtered document list
    if stats["sources"]:
        filtered_sources = stats["sources"]
        
        if filter_type == "PDF Files":
            filtered_sources = [s for s in stats["sources"] if s.lower().endswith('.pdf')]
        elif filter_type == "CSV Files":
            filtered_sources = [s for s in stats["sources"] if s.lower().endswith('.csv')]
        elif filter_type == "Text Files":
            filtered_sources = [s for s in stats["sources"] if s.lower().endswith(('.txt', '.md', '.markdown'))]
        
        if filtered_sources:
            st.markdown(f"**{filter_type} ({len(filtered_sources)} documents):**")
            for i, source in enumerate(filtered_sources, 1):
                st.markdown(f"""
                <div class="filter-item slide-in">
                    <strong>{i}.</strong> {source}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No documents found for filter: {filter_type}")

def sidebar_content():
    """Clean sidebar with file upload and controls"""
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'csv', 'txt', 'md', 'markdown'],
            accept_multiple_files=True,
            help="Supported formats: PDF, CSV, TXT, MD, Markdown",
            key="file_uploader"
        )
        
        if uploaded_files:
            st.markdown("### Uploaded Files")
            
            if st.button("Process Files", type="primary", use_container_width=True, key="process_files_btn"):
                with st.spinner("Processing uploaded files..."):
                    total_processed = 0
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        all_documents = []
                        
                        for uploaded_file in uploaded_files:
                            st.write(f"Processing: {uploaded_file.name}")
                            documents = process_uploaded_file(uploaded_file, temp_dir)
                            if documents:
                                all_documents.extend(documents)
                                total_processed += 1
                        
                        if all_documents:
                            chunks = split_documents(all_documents)
                            new_chunks_added = add_documents_to_chroma(chunks, st.session_state.db)
                            
                            if new_chunks_added > 0:
                                st.success(f"Successfully processed {total_processed} files ({new_chunks_added} new chunks added)!")
                                # Clear cached results
                                if hasattr(st.session_state, 'search_results'):
                                    del st.session_state.search_results
                                if hasattr(st.session_state, 'current_summary'):
                                    del st.session_state.current_summary
                            else:
                                st.info("All documents were already in the database")
                        else:
                            st.error("No documents could be processed")
            
            # Display file information
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getbuffer()) / 1024
                st.markdown(f"""
                <div class="file-info slide-in">
                    <strong>{uploaded_file.name}</strong><br>
                    <small>Size: {file_size:.1f} KB | Type: {uploaded_file.type}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", type="secondary", use_container_width=True, key="clear_chat_btn"):
                st.session_state.messages = []
                if hasattr(st.session_state, 'last_sources'):
                    del st.session_state.last_sources
                welcome_msg = "Chat cleared! Ready for new questions."
                st.session_state.messages.append(AIMessage(welcome_msg))
        
        with col2:
            if st.button("Export", type="secondary", use_container_width=True, key="export_btn"):
                stats = get_database_stats(st.session_state.db)
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "stats": stats,
                    "chat_history": [
                        {"type": "user" if isinstance(msg, HumanMessage) else "assistant", 
                         "content": msg.content}
                        for msg in st.session_state.messages
                    ]
                }
                
                st.download_button(
                    "Download",
                    str(export_data),
                    file_name=f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_export_btn"
                )
        
        # Database statistics in sidebar
        stats = get_database_stats(st.session_state.db)
        st.markdown("### Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Documents", stats["unique_sources"])
        with col2:
            create_metric_card("Chunks", stats["total_chunks"])

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Advanced RAG Assistant", 
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    load_css()
    
    # Create header
    create_custom_header()
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.db, st.session_state.model = initialize_rag_system()
                st.session_state.rag_initialized = True
                time.sleep(1)
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "👋 Welcome to Advanced RAG Assistant! I can chat, summarize, search, and analyze your documents. Upload files using the sidebar to get started."
        st.session_state.messages.append(AIMessage(welcome_msg))
    
    # Initialize active tab state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"
    
    # Sidebar content
    sidebar_content()
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Summarize", "Search", "Analytics"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        summarizer_interface()
    
    with tab3:
        search_interface()
    
    with tab4:
        analytics_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem;">
        Advanced RAG Assistant - Built with Streamlit & LangChain<br>
        Features: Intelligent Chat • Document Summarization • Advanced Search • Analytics
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()