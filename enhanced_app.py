
import streamlit as st
import sys
import os
import time
from typing import Dict, List, Any
import json
import tempfile
import shutil

# Page configuration
st.set_page_config(
    page_title="🤖 Enhanced Hybrid QnA Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced RAG system
try:
    from inference import InferenceAPI, get_api_instance
    from enhanced_retriever import EnhancedVectorRetriever, EnhancedWikipediaRetriever
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import RAG system: {e}")
    RAG_AVAILABLE = False
# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }

    .source-box {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .upload-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }

    .stats-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'api' not in st.session_state:
        st.session_state.api = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'enhanced_retriever' not in st.session_state:
        st.session_state.enhanced_retriever = None

def initialize_enhanced_system():
    """Initialize the enhanced RAG system"""
    if not RAG_AVAILABLE:
        st.error("❌ RAG system components not available. Please check your installation.")
        return False

    if st.session_state.api is None:
        with st.spinner("🔄 Initializing Enhanced RAG System..."):
            try:
                # Create enhanced retrievers
                vector_retriever = EnhancedVectorRetriever()
                wiki_retriever = EnhancedWikipediaRetriever()

                # Initialize with sample corpus
                sample_texts = [
                    """
                    Machine learning is a subset of artificial intelligence that focuses on algorithms 
                    that can learn and make decisions from data. It includes supervised learning, 
                    unsupervised learning, and reinforcement learning approaches. Deep learning, 
                    using neural networks, is a popular subset of machine learning.
                    """,
                    """
                    Natural Language Processing (NLP) is a field of AI that deals with the interaction 
                    between computers and humans using natural language. It includes tasks like 
                    sentiment analysis, named entity recognition, machine translation, and question answering.
                    """,
                    """
                    Python is a popular programming language for data science and machine learning. 
                    It has extensive libraries like NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch 
                    that make it easy to work with data and build ML models.
                    """,
                    f"""
                    Current US Politics 2024: Donald Trump was elected as the 47th President of the United States 
                    in the 2024 presidential election, defeating incumbent Vice President Kamala Harris. 
                    Trump previously served as the 45th President from 2017-2021. The election took place on 
                    November 5, 2024, and Trump is scheduled to take office on January 20, 2025.
                    """
                ]

                # Build initial index
                vector_retriever.build_index_from_texts(sample_texts)
                st.session_state.enhanced_retriever = vector_retriever

                # Create hybrid retriever
                from retriever import HybridRetriever
                hybrid_retriever = HybridRetriever(
                    vector_retriever=vector_retriever,
                    external_retrievers=[wiki_retriever]
                )
                # Create API with enhanced retriever
                from rag_pipeline import RAGPipeline
                from qa_model import ExtractiveQAModel
                pipeline = RAGPipeline(
                    retriever=hybrid_retriever,
                    qa_model=ExtractiveQAModel(),
                    top_k_retrieval=5,
                    max_context_length=1000,
                    min_confidence=0.3
                )

                st.session_state.api = InferenceAPI()
                st.session_state.api.pipeline = pipeline
                st.session_state.api.is_initialized = True

                st.session_state.system_initialized = True
                st.success("✅ Enhanced RAG System initialized successfully!")
                return True

            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")
                return False

    return st.session_state.system_initialized

def render_document_upload():
    """Render document upload interface"""
    st.header("📁 Document Upload")

    st.markdown("""
    <div class="upload-box">
        <strong>📚 Upload Your Documents</strong><br>
        Upload PDF, DOCX, or TXT files to add them to the knowledge base.
        Your documents will be processed and made searchable immediately.
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files and st.session_state.enhanced_retriever:
        if st.button("🚀 Process Uploaded Documents", type="primary"):
            process_uploaded_documents(uploaded_files)

    # Show uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("📋 Uploaded Documents")

        for doc_info in st.session_state.uploaded_docs:
            with st.expander(f"📄 {doc_info['name']} ({doc_info['size']} chars)"):
                st.write(f"**Uploaded:** {doc_info['upload_time']}")
                st.write(f"**Chunks:** {doc_info['chunks']}")
                st.write(f"**Preview:** {doc_info['preview'][:200]}...")

def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and add to retriever"""
    if not st.session_state.enhanced_retriever:
        st.error("❌ Enhanced retriever not initialized")
        return

    with st.spinner("🔄 Processing uploaded documents..."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        try:
            # Save uploaded files temporarily
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_path)

            # Add documents to retriever
            num_added = st.session_state.enhanced_retriever.add_documents_from_files(file_paths)

            if num_added > 0:
                # Update session state
                for i, uploaded_file in enumerate(uploaded_files):
                    # Read content for preview
                    try:
                        content = st.session_state.enhanced_retriever._extract_text_from_file(file_paths[i])
                        doc_info = {
                            'name': uploaded_file.name,
                            'size': len(content),
                            'upload_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'chunks': len(st.session_state.enhanced_retriever.chunk_text(content)),
                            'preview': content[:500]
                        }
                        st.session_state.uploaded_docs.append(doc_info)
                    except:
                        pass

                st.success(f"✅ Successfully processed {num_added} documents!")
                st.info("💡 Your documents are now searchable. Try asking questions about their content!")

            else:
                st.warning("⚠️ No documents were processed. Please check the file formats.")

        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")

        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def render_enhanced_sidebar():
    """Render enhanced sidebar with upload stats"""
    st.sidebar.header("⚙️ Enhanced Settings")

    # Document upload summary
    st.sidebar.subheader("📚 Knowledge Base")
    if st.session_state.uploaded_docs:
        total_docs = len(st.session_state.uploaded_docs)
        total_chars = sum(doc['size'] for doc in st.session_state.uploaded_docs)
        st.sidebar.metric("Uploaded Documents", total_docs)
        st.sidebar.metric("Total Content", f"{total_chars:,} chars")
    else:
        st.sidebar.info("No documents uploaded yet")

    # System settings
    st.sidebar.subheader("🔧 RAG Configuration")
    top_k = st.sidebar.slider("Documents to retrieve", 3, 10, 5, key='top_k')
    max_context = st.sidebar.slider("Max context length", 500, 2000, 1000, step=100, key='max_context')
    min_confidence = st.sidebar.slider("Min confidence threshold", 0.1, 0.8, 0.3, step=0.1, key='min_confidence')

    # System information
    st.sidebar.subheader("📊 System Info")
    if st.session_state.system_initialized and st.session_state.api:
        try:
            stats = st.session_state.api.get_stats()
            st.sidebar.metric("Total Requests", stats['statistics']['total_requests'])

            if stats['statistics']['total_requests'] > 0:
                success_rate = (stats['statistics']['successful_requests'] / 
                              stats['statistics']['total_requests'] * 100)
                st.sidebar.metric("Success Rate", f"{success_rate:.1f}%")
                st.sidebar.metric("Avg Response Time", 
                    f"{stats['statistics']['avg_processing_time']:.2f}s")

        except Exception as e:
            st.sidebar.error(f"Error loading stats: {e}")

    # Enhanced sample questions
    st.sidebar.subheader("💡 Sample Questions")
    sample_questions = [
        "Who is the current president of the United States?",
        "What is machine learning?", 
        "What programming language is popular for data science?",
        "Tell me about recent US election results",
        "How does artificial intelligence work?",
        "What are neural networks?",
        "What documents did I upload?"  # New question for uploaded docs
    ]

    for question in sample_questions:
        if st.sidebar.button(f"❓ {question[:35]}...", key=f"sample_{hash(question)}"):
            st.session_state.current_question = question

def render_enhanced_main_interface():
    """Render enhanced main interface"""
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced Hybrid QnA Agent</h1>', unsafe_allow_html=True)

    # Description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload documents, ask questions, get accurate answers with sources! 
            Enhanced with current information retrieval and document processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize system if needed
    if not st.session_state.system_initialized:
        if st.button("🚀 Initialize Enhanced RAG System", type="primary"):
            initialize_enhanced_system()
        return

    # Question input with enhanced functionality
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., Who is the current US president? or What did I upload about machine learning?",
            key="question_input"
        )

    with col2:
        ask_button = st.button("🔍 Ask", type="primary", disabled=not question.strip())

    # Show current knowledge base status
    if st.session_state.enhanced_retriever and hasattr(st.session_state.enhanced_retriever, 'index'):
        if st.session_state.enhanced_retriever.index:
            total_vectors = st.session_state.enhanced_retriever.index.ntotal
            st.info(f"📊 Knowledge base contains {total_vectors:,} text chunks from {len(st.session_state.uploaded_docs) + 4} documents")

    # Process question
    if ask_button and question.strip():
        process_enhanced_question(question.strip())

    # Display chat history (same as before but with enhanced display)
    render_enhanced_chat_history()

def process_enhanced_question(question: str):
    """Process question with enhanced features"""
    if not st.session_state.api:
        st.error("❌ System not initialized")
        return

    # Show processing with more details
    with st.spinner(f"🔄 Processing: {question}..."):
        start_time = time.time()

        try:
            # Get response from API
            response = st.session_state.api.ask(question)
            processing_time = time.time() - start_time

            # Enhanced response processing
            if response['status'] == 'success':
                # Check if this was answered from uploaded documents
                has_uploaded_content = any(
                    source.get('metadata', {}).get('type') == 'uploaded_file' 
                    for source in response['sources']
                )

                if has_uploaded_content:
                    st.success("✅ Found relevant information in your uploaded documents!")

            # Add to chat history
            chat_entry = {
                'question': question,
                'response': response,
                'timestamp': time.time(),
                'processing_time': processing_time,
                'has_uploaded_content': has_uploaded_content if response['status'] == 'success' else False
            }

            st.session_state.chat_history.append(chat_entry)

            # Clear the question input
            if 'current_question' in st.session_state:
                del st.session_state.current_question

        except Exception as e:
            st.error(f"❌ Error processing question: {e}")

def render_enhanced_chat_history():
    """Render enhanced chat history"""
    if not st.session_state.chat_history:
        st.info("💬 Ask a question to get started! Try asking about current events or your uploaded documents.")
        return

    st.header("💬 Conversation History")

    # Reverse order to show latest first
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        response = entry['response']

        # Question
        st.markdown(f"""
        <div class="question-box">
            <strong>❓ Question:</strong><br>
            {entry['question']}
        </div>
        """, unsafe_allow_html=True)

        # Answer with enhanced indicators
        if response['status'] == 'success':
            confidence_color = "🟢" if response['confidence'] > 0.7 else "🟡" if response['confidence'] > 0.4 else "🔴"

            # Add indicators for special content
            indicators = []
            if entry.get('has_uploaded_content'):
                indicators.append("📁 From uploaded docs")
            if any('wikipedia' in source.get('source', '') for source in response['sources']):
                indicators.append("🌐 From Wikipedia")

            indicator_text = " • ".join(indicators)

            st.markdown(f"""
            <div class="answer-box">
                <strong>🤖 Answer:</strong> {confidence_color} <em>(Confidence: {response['confidence']:.2f})</em><br>
                {response['answer']}<br>
                {f"<small style='color: #666;'>{indicator_text}</small>" if indicator_text else ""}
            </div>
            """, unsafe_allow_html=True)

            # Enhanced sources display
            if response['sources']:
                st.markdown("**📚 Sources:**")

                for j, source in enumerate(response['sources'][:3], 1):
                    source_type = "📁 Uploaded" if source.get('metadata', {}).get('type') == 'uploaded_file' else "🌐 Wikipedia"

                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{source_type} Source {j}</strong> ({source['source']}) - Score: {source['score']:.3f}<br>
                        <em>{source['content'][:200]}{'...' if len(source['content']) > 200 else ''}</em>
                    </div>
                    """, unsafe_allow_html=True)

            # Processing stats
            st.markdown(f"""
            <div style="text-align: right; color: #666; font-size: 0.8rem;">
                ⏱️ Processed in {response['processing_time']:.2f}s • 
                📊 {len(response['sources'])} sources • 
                🔧 {response['metadata'].get('qa_model_type', 'unknown')} model
            </div>
            """, unsafe_allow_html=True)

        else:
            # Error response
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Error:</strong><br>
                {response.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

def main():
    """Enhanced main application function"""
    # Initialize session state
    initialize_session_state()

    # Enhanced sidebar
    render_enhanced_sidebar()

    # Main tabs with document upload
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Upload Documents", "📦 Batch Processing", "📊 System Health"])

    with tab1:
        render_enhanced_main_interface()

    with tab2:
        render_document_upload()

    with tab3:
        render_batch_processing()

    with tab4:
        render_system_health()

def render_batch_processing():
    """Render batch processing interface (same as before)"""
    st.header("📦 Batch Processing")

    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first")
        return

    st.markdown("Process multiple questions at once:")

    questions_text = st.text_area(
        "Enter questions (one per line):",
        placeholder="Who is the US president?\nWhat is machine learning?\nWhat did I upload about AI?",
        height=150
    )

    if st.button("🚀 Process Batch", type="primary"):
        if questions_text.strip():
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

            if questions:
                with st.spinner(f"Processing {len(questions)} questions..."):
                    try:
                        responses = st.session_state.api.batch_ask(questions)

                        st.success(f"✅ Processed {len(responses)} questions!")

                        results_data = []
                        for q, r in zip(questions, responses):
                            results_data.append({
                                'Question': q,
                                'Answer': r['answer'][:100] + '...' if len(r['answer']) > 100 else r['answer'],
                                'Confidence': f"{r['confidence']:.2f}",
                                'Sources': len(r['sources']),
                                'Status': r['status'],
                                'Time (s)': f"{r['processing_time']:.2f}"
                            })

                        st.dataframe(results_data, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {e}")

def render_system_health():
    """Render system health (same as before)"""
    st.header("🏥 System Health")

    if not RAG_AVAILABLE:
        st.error("❌ RAG system components not available")
        return

    if st.session_state.api:
        try:
            health = st.session_state.api.health_check()

            if health['status'] == 'healthy':
                st.success("✅ System is healthy and ready")
            else:
                st.warning(f"⚠️ System status: {health['status']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Pipeline Status", "✅ Ready" if health['pipeline_ready'] else "❌ Not Ready")

            with col2:
                st.metric("Version", health.get('version', 'Unknown'))

            with col3:
                st.metric("Documents", len(st.session_state.uploaded_docs))

            # Enhanced stats
            if st.session_state.api:
                stats = st.session_state.api.get_stats()
                st.json(stats)

        except Exception as e:
            st.error(f"❌ Health check failed: {e}")

    else:
        st.info("ℹ️ System not initialized. Click 'Initialize Enhanced RAG System' to start.")

        if st.button("🔄 Initialize Now"):
            initialize_enhanced_system()

if __name__ == "__main__":
    main()
