
import os
import sys
import logging
from typing import Dict, List, Optional, Any
import json
import time
from dataclasses import asdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline, RAGResponse, create_simple_rag_pipeline
from retriever import VectorRetriever, WikipediaRetriever, HybridRetriever
from qa_model import ExtractiveQAModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceAPI:
    """
    Single API endpoint for the RAG system
    Handles initialization, caching, and request processing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the inference API

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.pipeline = None
        self.is_initialized = False
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        }

        logger.info("InferenceAPI initialized with config")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'use_sample_corpus': True,
            'retriever_type': 'hybrid',  # 'vector', 'wikipedia', or 'hybrid'
            'qa_model_type': 'extractive',  # 'extractive', 'generative', or 'hybrid'
            'top_k_retrieval': 5,
            'max_context_length': 1000,
            'min_confidence': 0.3,
            'enable_wikipedia': True,
            'cache_models': True
        }

    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline
        Returns True if successful, False otherwise
        """
        if self.is_initialized:
            logger.info("Pipeline already initialized")
            return True

        try:
            logger.info("Initializing RAG pipeline...")
            start_time = time.time()

            # Create pipeline based on configuration
            if self.config['use_sample_corpus']:
                logger.info("Using sample corpus for demonstration")
                self.pipeline = create_simple_rag_pipeline(
                    top_k_retrieval=self.config['top_k_retrieval'],
                    max_context_length=self.config['max_context_length'],
                    min_confidence=self.config['min_confidence']
                )
            else:
                # Custom pipeline setup
                retriever = self._create_retriever()
                qa_model = self._create_qa_model()

                self.pipeline = RAGPipeline(
                    retriever=retriever,
                    qa_model=qa_model,
                    top_k_retrieval=self.config['top_k_retrieval'],
                    max_context_length=self.config['max_context_length'],
                    min_confidence=self.config['min_confidence']
                )

            init_time = time.time() - start_time
            self.is_initialized = True

            logger.info(f"RAG pipeline initialized successfully in {init_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False

    def _create_retriever(self):
        """Create retriever based on configuration"""
        if self.config['retriever_type'] == 'vector':
            return VectorRetriever()
        elif self.config['retriever_type'] == 'wikipedia':
            return WikipediaRetriever()
        else:  # hybrid
            vector_retriever = VectorRetriever()
            external_retrievers = []
            if self.config['enable_wikipedia']:
                external_retrievers.append(WikipediaRetriever())

            return HybridRetriever(
                vector_retriever=vector_retriever,
                external_retrievers=external_retrievers
            )

    def _create_qa_model(self):
        """Create QA model based on configuration"""
        if self.config['qa_model_type'] == 'extractive':
            return ExtractiveQAModel()
        else:
            # Add other model types here as needed
            logger.warning(f"Unknown QA model type: {self.config['qa_model_type']}, using extractive")
            return ExtractiveQAModel()

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Main inference method - ask a question and get an answer

        Args:
            question: The question to answer
            **kwargs: Additional parameters

        Returns:
            Dictionary containing the response
        """
        # Ensure pipeline is initialized
        if not self.is_initialized:
            if not self.initialize():
                return self._create_error_response("Failed to initialize pipeline")

        # Validate input
        if not question or not question.strip():
            return self._create_error_response("Question cannot be empty")

        # Update statistics
        self.stats['total_requests'] += 1

        try:
            logger.info(f"Processing question: {question[:50]}...")
            start_time = time.time()

            # Process the question
            response = self.pipeline.process_question(question, **kwargs)

            # Update statistics
            processing_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['successful_requests']
            )

            # Convert to API response format
            api_response = self._rag_response_to_dict(response)
            api_response['status'] = 'success'

            logger.info(f"Question processed successfully in {processing_time:.2f}s")
            return api_response

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            self.stats['failed_requests'] += 1
            return self._create_error_response(str(e))

    def batch_ask(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch

        Args:
            questions: List of questions
            **kwargs: Additional parameters

        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions...")

        responses = []
        for question in questions:
            response = self.ask(question, **kwargs)
            responses.append(response)

        logger.info(f"Batch processing completed")
        return responses

    def _rag_response_to_dict(self, response: RAGResponse) -> Dict[str, Any]:
        """Convert RAGResponse to dictionary for API"""
        result = {
            'question': response.question,
            'answer': response.answer,
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'sources': [
                {
                    'content': source.content[:200] + '...' if len(source.content) > 200 else source.content,
                    'source': source.source,
                    'score': source.score,
                    'metadata': source.metadata
                }
                for source in response.sources
            ],
            'metadata': response.metadata
        }
        return result

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'status': 'error',
            'error': error_message,
            'question': '',
            'answer': '',
            'confidence': 0.0,
            'sources': [],
            'processing_time': 0.0,
            'metadata': {'error': True}
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            'pipeline_initialized': self.is_initialized,
            'configuration': self.config,
            'statistics': self.stats.copy(),
            'pipeline_info': self.pipeline.get_pipeline_info() if self.is_initialized else None
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            'status': 'healthy' if self.is_initialized else 'not_initialized',
            'timestamp': time.time(),
            'pipeline_ready': self.is_initialized,
            'version': '1.0.0'
        }

# Global API instance for easy access
api = None

def get_api_instance(config: Optional[Dict] = None) -> InferenceAPI:
    """Get or create global API instance"""
    global api
    if api is None:
        api = InferenceAPI(config)
    return api

def ask_question(question: str, **kwargs) -> Dict[str, Any]:
    """Simple function interface for asking questions"""
    api_instance = get_api_instance()
    return api_instance.ask(question, **kwargs)

def main_demo():
    """
    Demo function showing the inference API in action
    """
    print("🚀 RAG Inference API Demo")
    print("="*50)

    # Initialize API
    print("\n1. Initializing Inference API...")
    api_instance = get_api_instance()

    # Health check
    health = api_instance.health_check()
    print(f"Health Status: {health['status']}")

    # Initialize pipeline
    print("\n2. Initializing RAG Pipeline...")
    success = api_instance.initialize()
    if not success:
        print("❌ Failed to initialize pipeline")
        return

    print("✅ Pipeline initialized successfully!")

    # Sample questions
    questions = [
        "What is machine learning?",
        "What programming language is popular for data science?",
        "How does deep learning work?",
        "What are the applications of NLP?"
    ]

    print(f"\n3. Processing {len(questions)} questions...")

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")

        # Ask the question
        response = api_instance.ask(question)

        if response['status'] == 'success':
            print(f"A: {response['answer']}")
            print(f"Confidence: {response['confidence']:.3f}")
            print(f"Sources: {len(response['sources'])}")
            print(f"Processing Time: {response['processing_time']:.2f}s")

            # Show top source
            if response['sources']:
                top_source = response['sources'][0]
                print(f"Top Source: {top_source['source']} (score: {top_source['score']:.3f})")
        else:
            print(f"❌ Error: {response['error']}")

    # Show statistics
    print("\n" + "="*50)
    print("📊 API Statistics:")
    stats = api_instance.get_stats()
    print(f"Total Requests: {stats['statistics']['total_requests']}")
    print(f"Successful: {stats['statistics']['successful_requests']}")
    print(f"Failed: {stats['statistics']['failed_requests']}")
    print(f"Average Processing Time: {stats['statistics']['avg_processing_time']:.2f}s")

    # Test batch processing
    print("\n4. Testing Batch Processing...")
    batch_questions = [
        "What is AI?",
        "What is Python?"
    ]

    batch_responses = api_instance.batch_ask(batch_questions)
    print(f"✅ Processed {len(batch_responses)} questions in batch")

    for i, response in enumerate(batch_responses):
        if response['status'] == 'success':
            print(f"  Q{i+1}: {response['answer'][:50]}... (conf: {response['confidence']:.2f})")

    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main_demo()
