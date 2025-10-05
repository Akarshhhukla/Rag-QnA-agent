
import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import time
from dataclasses import dataclass, asdict
import json

# Import our custom modules
from retriever import VectorRetriever, WikipediaRetriever, HybridRetriever, RetrievedDocument
from qa_model import ExtractiveQAModel, QAResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Complete response from RAG pipeline"""
    question: str
    answer: str
    confidence: float
    sources: List[RetrievedDocument]
    processing_time: float
    metadata: Dict[str, Any]

class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline
    Combines document retrieval with question answering
    """

    def __init__(self,
                 retriever: Optional[Any] = None,
                 qa_model: Optional[Any] = None,
                 top_k_retrieval: int = 5,
                 max_context_length: int = 1000,
                 min_confidence: float = 0.3):
        """
        Initialize RAG pipeline

        Args:
            retriever: Document retriever (VectorRetriever, HybridRetriever, etc.)
            qa_model: QA model (ExtractiveQAModel, GenerativeQAModel, etc.)
            top_k_retrieval: Number of documents to retrieve
            max_context_length: Maximum context length for QA model
            min_confidence: Minimum confidence threshold for answers
        """

        self.top_k_retrieval = top_k_retrieval
        self.max_context_length = max_context_length
        self.min_confidence = min_confidence

        # Initialize retriever
        if retriever is None:
            logger.info("No retriever provided, initializing default HybridRetriever...")
            vector_retriever = VectorRetriever()
            wiki_retriever = WikipediaRetriever()
            self.retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                external_retrievers=[wiki_retriever]
            )
        else:
            self.retriever = retriever

        # Initialize QA model
        if qa_model is None:
            logger.info("No QA model provided, initializing default ExtractiveQAModel...")
            self.qa_model = ExtractiveQAModel()
        else:
            self.qa_model = qa_model

        logger.info("RAG Pipeline initialized successfully")
        logger.info(f"Configuration: top_k={top_k_retrieval}, max_context={max_context_length}, min_conf={min_confidence}")

    def process_question(self, question: str, **kwargs) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve documents + answer question

        Args:
            question: User's question
            **kwargs: Additional parameters (e.g., top_k override)

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        logger.debug(f"Retrieving documents for: {question}")
        top_k = kwargs.get('top_k', self.top_k_retrieval)

        try:
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return self._create_error_response(question, "Retrieval failed", start_time)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return self._create_no_results_response(question, start_time)

        # Step 2: Combine and truncate context
        combined_context = self._combine_contexts(retrieved_docs)

        # Step 3: Answer the question
        logger.debug(f"Answering question with context length: {len(combined_context)}")

        try:
            qa_result = self.qa_model.answer_question(question, combined_context)
            logger.debug(f"QA result: {qa_result.answer[:50]}... (conf: {qa_result.confidence:.3f})")
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return self._create_error_response(question, "QA failed", start_time)

        # Step 4: Post-process and validate
        final_response = self._create_final_response(
            question=question,
            qa_result=qa_result,
            sources=retrieved_docs,
            start_time=start_time
        )

        processing_time = time.time() - start_time
        logger.info(f"Question processed in {processing_time:.2f}s (conf: {final_response.confidence:.3f})")

        return final_response

    def _combine_contexts(self, docs: List[RetrievedDocument]) -> str:
        """
        Combine multiple document contexts into single context for QA
        """
        combined = ""
        current_length = 0

        for doc in docs:
            doc_text = doc.content.strip()
            doc_length = len(doc_text)

            # Check if adding this document would exceed max length
            if current_length + doc_length > self.max_context_length:
                # Try to fit partial document
                remaining_space = self.max_context_length - current_length
                if remaining_space > 100:  # Only if we have reasonable space
                    doc_text = doc_text[:remaining_space] + "..."
                    combined += "\n\n" + doc_text
                break

            combined += "\n\n" + doc_text
            current_length += doc_length + 4  # +4 for newlines

        return combined.strip()

    def _create_final_response(self, 
                              question: str, 
                              qa_result: QAResult, 
                              sources: List[RetrievedDocument],
                              start_time: float) -> RAGResponse:
        """Create the final RAG response"""

        processing_time = time.time() - start_time

        # Enhance answer if confidence is low
        if qa_result.confidence < self.min_confidence:
            enhanced_answer = self._enhance_low_confidence_answer(qa_result, sources)
        else:
            enhanced_answer = qa_result.answer

        # Create metadata
        metadata = {
            "qa_model_type": qa_result.model_type,
            "num_sources": len(sources),
            "context_length": len(qa_result.source_text),
            "retrieval_scores": [doc.score for doc in sources],
            "source_types": list(set([doc.source.split(':')[0] for doc in sources])),
            "processing_steps": [
                "document_retrieval",
                "context_combination", 
                "question_answering",
                "response_enhancement"
            ]
        }

        # Add QA-specific metadata
        if qa_result.metadata:
            metadata.update(qa_result.metadata)

        return RAGResponse(
            question=question,
            answer=enhanced_answer,
            confidence=qa_result.confidence,
            sources=sources,
            processing_time=processing_time,
            metadata=metadata
        )

    def _enhance_low_confidence_answer(self, 
                                     qa_result: QAResult, 
                                     sources: List[RetrievedDocument]) -> str:
        """
        Enhance low-confidence answers with additional context
        """
        if qa_result.answer == "No answer found" or not qa_result.answer.strip():
            # Extract key information from top source
            if sources:
                top_source = sources[0]
                sentences = top_source.content.split('.')
                if sentences:
                    return f"Based on available information: {sentences[0].strip()}."

            return "I couldn't find a specific answer in the available sources."

        # Add confidence indicator for low-confidence answers
        if qa_result.confidence < self.min_confidence:
            return f"{qa_result.answer} (Note: This answer has lower confidence - please verify from sources.)"

        return qa_result.answer

    def _create_error_response(self, question: str, error_msg: str, start_time: float) -> RAGResponse:
        """Create error response"""
        return RAGResponse(
            question=question,
            answer=f"Sorry, I encountered an error: {error_msg}",
            confidence=0.0,
            sources=[],
            processing_time=time.time() - start_time,
            metadata={"error": error_msg, "status": "error"}
        )

    def _create_no_results_response(self, question: str, start_time: float) -> RAGResponse:
        """Create response when no documents are retrieved"""
        return RAGResponse(
            question=question,
            answer="I couldn't find relevant information to answer your question.",
            confidence=0.0,
            sources=[],
            processing_time=time.time() - start_time,
            metadata={"status": "no_results", "reason": "no_documents_retrieved"}
        )

    def batch_process(self, questions: List[str], **kwargs) -> List[RAGResponse]:
        """
        Process multiple questions in batch
        """
        logger.info(f"Processing batch of {len(questions)} questions...")
        responses = []

        for i, question in enumerate(questions):
            logger.debug(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            response = self.process_question(question, **kwargs)
            responses.append(response)

        logger.info(f"Batch processing completed. Average confidence: {sum(r.confidence for r in responses) / len(responses):.3f}")
        return responses

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "retriever_type": type(self.retriever).__name__,
            "qa_model_type": type(self.qa_model).__name__,
            "top_k_retrieval": self.top_k_retrieval,
            "max_context_length": self.max_context_length,
            "min_confidence": self.min_confidence
        }

# Utility functions for easy setup
def create_simple_rag_pipeline(**kwargs) -> RAGPipeline:
    """
    Create a simple RAG pipeline with default components
    """
    logger.info("Creating simple RAG pipeline...")

    # Create sample corpus if no retriever provided
    if 'retriever' not in kwargs:
        logger.info("Setting up sample corpus...")
        sample_texts = [
            """
            Machine learning is a subset of artificial intelligence that focuses on algorithms 
            that can learn and make decisions from data. It includes supervised learning, 
            unsupervised learning, and reinforcement learning approaches.
            """,
            """
            Natural Language Processing (NLP) is a field of AI that deals with the interaction 
            between computers and humans using natural language. It includes tasks like 
            sentiment analysis, named entity recognition, and machine translation.
            """,
            """
            Deep learning is a subset of machine learning that uses neural networks with 
            multiple layers to model and understand complex patterns in data. It has been 
            particularly successful in computer vision and natural language processing.
            """,
            """
            Python is a popular programming language for data science and machine learning. 
            It has extensive libraries like NumPy, Pandas, Scikit-learn, and TensorFlow 
            that make it easy to work with data and build ML models.
            """
        ]

        # Create vector retriever with sample data
        vector_retriever = VectorRetriever()
        vector_retriever.build_index_from_texts(sample_texts)

        # Create hybrid retriever
        wiki_retriever = WikipediaRetriever()
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            external_retrievers=[wiki_retriever]
        )

        kwargs['retriever'] = retriever

    return RAGPipeline(**kwargs)

def demo_rag_pipeline():
    """
    Demo function showing RAG pipeline in action
    """
    print("🚀 RAG Pipeline Demo")
    print("="*50)

    # Create pipeline
    print("\n1. Initializing RAG pipeline...")
    pipeline = create_simple_rag_pipeline()

    # Sample questions
    questions = [
        "What is machine learning?",
        "What programming language is popular for data science?", 
        "What are neural networks used for?",
        "How does supervised learning work?",
    ]

    print(f"\n2. Processing {len(questions)} questions...")

    # Process each question
    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {question}")

        try:
            response = pipeline.process_question(question)

            print(f"A: {response.answer}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Sources: {len(response.sources)} documents")
            print(f"Processing time: {response.processing_time:.2f}s")

            # Show top source
            if response.sources:
                top_source = response.sources[0]
                print(f"Top source: {top_source.source} (score: {top_source.score:.3f})")
                print(f"Excerpt: {top_source.content[:100]}...")

        except Exception as e:
            print(f"❌ Error processing question: {e}")

    # Show pipeline info
    print("\n" + "="*50)
    print("📊 Pipeline Information:")
    info = pipeline.get_pipeline_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_rag_pipeline()
