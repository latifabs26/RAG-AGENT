import streamlit as st
import tempfile
import time
from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from rag_utils import (
    initialize_rag_system,
    process_uploaded_file,
    split_documents,
    add_documents_to_chroma,
    query_rag_with_history,
    generate_summary,
    search_documents,
    get_database_stats,
    load_css,
    create_custom_header,
    create_metric_card,
    display_sources_beautifully,
    display_search_results
)

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
        st.markdown(f"**Search Results for:** *{st.session_state.search_query_used}*")
        display_search_results(st.session_state.search_results)
        
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
        import pandas as pd
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