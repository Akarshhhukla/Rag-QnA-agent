import os
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Core ML libraries
import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    pipeline
)

# For confidence scoring
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Data class for QA model results"""
    answer: str
    confidence: float
    source_text: str
    model_type: str  # 'extractive' or 'generative'
    start_pos: Optional[int] = None  # For extractive answers
    end_pos: Optional[int] = None
    metadata: Optional[Dict] = None

class BaseQAModel:
    """Base class for QA models"""

    def answer_question(self, question: str, context: str) -> QAResult:
        raise NotImplementedError("Subclasses must implement answer_question method")

    def batch_answer(self, questions: List[str], contexts: List[str]) -> List[QAResult]:
        """Answer multiple questions (default implementation)"""
        results = []
        for q, c in zip(questions, contexts):
            results.append(self.answer_question(q, c))
        return results

class ExtractiveQAModel(BaseQAModel):
    """
    Extractive QA using DistilBERT (finds exact spans in the text)
    """

    def __init__(self, 
                 model_name: str = "distilbert/distilbert-base-uncased-distilled-squad",
                 max_length: int = 512,
                 device: str = None):

        self.model_name = model_name
        self.max_length = max_length
        self.max_answer_length = 30
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading extractive QA model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Extractive QA model loaded successfully")

    def answer_question(self, question: str, context: str) -> QAResult:
        """
        Extract answer span from context for the given question
        """
        # Tokenize input with offset mapping for accurate span extraction
        inputs = self.tokenizer(
            question, 
            context,
            max_length=self.max_length,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # Extract offset mapping before moving to device
        offset_mapping = inputs.pop("offset_mapping")[0]

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get start and end logits
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

        # Find best valid span with constraints
        best_span, confidence = self._find_best_span(
            start_logits, 
            end_logits, 
            self.max_answer_length
        )
        
        start_idx, end_idx = best_span

        # Extract answer using offset mapping for clean text
        input_ids = inputs['input_ids'][0].cpu()
        
        # Use offset mapping to get character positions in original context
        if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
            start_char = offset_mapping[start_idx][0].item()
            end_char = offset_mapping[end_idx][1].item()
            answer = context[start_char:end_char].strip()
        else:
            # Fallback to token decoding
            answer_tokens = input_ids[start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Handle empty answers
        if not answer.strip():
            answer = "No answer found"
            confidence = 0.0

        return QAResult(
            answer=answer,
            confidence=confidence,
            source_text=context,
            model_type="extractive",
            start_pos=start_idx,
            end_pos=end_idx,
            metadata={
                "question": question,
                "model_name": self.model_name
            }
        )

    def _find_best_span(self, start_logits: torch.Tensor, end_logits: torch.Tensor, 
                        max_answer_length: int) -> Tuple[Tuple[int, int], float]:
        """
        Find the best answer span with proper constraints and joint scoring
        """
        # Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        # Find top-k start and end positions
        top_k = min(20, len(start_logits))
        top_start_indices = torch.topk(start_logits, top_k).indices
        top_end_indices = torch.topk(end_logits, top_k).indices

        # Find best valid span
        best_score = 0.0
        best_span = (0, 0)

        for start_idx in top_start_indices:
            for end_idx in top_end_indices:
                # Validate span constraints
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > max_answer_length:
                    continue

                # Joint probability score
                score = (start_probs[start_idx] * end_probs[end_idx]).item()
                
                if score > best_score:
                    best_score = score
                    best_span = (start_idx.item(), end_idx.item())

        # If no valid span found, use simple argmax with correction
        if best_score == 0.0:
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()
            if end_idx < start_idx:
                end_idx = start_idx
            if end_idx - start_idx + 1 > max_answer_length:
                end_idx = start_idx + max_answer_length - 1
            best_span = (start_idx, end_idx)
            best_score = (start_probs[start_idx] * end_probs[end_idx]).item()

        return best_span, float(best_score)

class GenerativeQAModel(BaseQAModel):
    """
    Generative QA using T5 (generates new text as answers)
    """

    def __init__(self, 
                 model_name: str = "t5-small",
                 max_input_length: int = 512,
                 max_output_length: int = 64,  # Reduced to prevent hanging
                 device: str = None):

        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading generative QA model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model - FIXED: Use AutoTokenizer for all models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        
        if "t5" in model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        logger.info("Generative QA model loaded successfully")

    def answer_question(self, question: str, context: str) -> QAResult:
        """
        Generate answer for the given question and context
        """
        try:
            # Format input for T5 (question answering task)
            if "t5" in self.model_name.lower():
                input_text = f"question: {question} context: {context}"
            else:
                input_text = f"Question: {question}\nContext: {context}\nAnswer:"

            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate answer with timeout protection
            logger.debug(f"Generating answer for: {question[:50]}...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_output_length,  # Changed from max_length
                    min_new_tokens=1,  # Added: ensure at least 1 token
                    num_beams=2,  # Reduced from 4 for speed
                    early_stopping=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.tokenizer.eos_token_id  # Added: force ending
                )

            # Decode generated answer
            generated_ids = outputs.sequences[0]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Clean up answer
            answer = answer.strip()
            
            # Handle empty or invalid answers
            if not answer or len(answer) < 2:
                answer = "Unable to generate answer"
                confidence = 0.0
            else:
                # Compute confidence from generation scores
                confidence = self._estimate_confidence(answer, question, context, outputs.scores, generated_ids)

            logger.debug(f"Generated answer: {answer[:50]}...")

            return QAResult(
                answer=answer,
                confidence=confidence,
                source_text=context,
                model_type="generative",
                metadata={
                    "question": question,
                    "model_name": self.model_name,
                    "input_length": len(input_text.split())
                }
            )
            
        except Exception as e:
            logger.error(f"Error in generative QA: {e}")
            return QAResult(
                answer=f"Error: {str(e)}",
                confidence=0.0,
                source_text=context,
                model_type="generative",
                metadata={
                    "question": question,
                    "model_name": self.model_name,
                    "error": str(e)
                }
            )

    def _estimate_confidence(self, answer: str, question: str, context: str, 
                            scores=None, generated_ids=None) -> float:
        """
        Estimate confidence score for generated answers
        """
        # Try to use generation scores if available
        if scores is not None and generated_ids is not None:
            try:
                return self._compute_generation_confidence(scores, generated_ids)
            except:
                pass  # Fall back to heuristic

        # Fallback to heuristic confidence estimation
        if not answer.strip():
            return 0.0

        # Basic heuristics for confidence
        score = 0.5  # Base score

        # Length heuristic (not too short, not too long)
        answer_length = len(answer.split())
        if 3 <= answer_length <= 20:
            score += 0.2
        elif answer_length < 3:
            score -= 0.2

        # Check if answer contains question words (might indicate confusion)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0

        if overlap > 0.5:  # Too much overlap might indicate echoing
            score -= 0.1

        # Check if answer seems relevant to context
        context_words = set(context.lower().split())
        context_overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0

        if context_overlap > 0.3:  # Good context relevance
            score += 0.2

        # Clamp between 0 and 1
        return max(0.0, min(1.0, score))

    def _compute_generation_confidence(self, scores: Tuple[torch.Tensor], 
                                       generated_ids: torch.Tensor) -> float:
        """
        Compute confidence from generation scores using token-level probabilities
        """
        if scores is None or len(scores) == 0:
            return 0.5  # Default if scores unavailable

        # Compute average log probability per token
        log_probs = []
        for i, score in enumerate(scores):
            if i + 1 < len(generated_ids):
                token_id = generated_ids[i + 1]
                probs = F.softmax(score[0], dim=0)
                if token_id < len(probs):
                    log_probs.append(torch.log(probs[token_id] + 1e-10).item())

        if not log_probs:
            return 0.5

        # Average log probability, normalized to [0, 1] range
        avg_log_prob = sum(log_probs) / len(log_probs)
        
        # Convert to pseudo-probability (heuristic mapping)
        confidence = max(0.0, min(1.0, (avg_log_prob + 10) / 10))
        
        return confidence

class HybridQAModel(BaseQAModel):
    """
    Hybrid QA that combines extractive and generative approaches
    """

    def __init__(self, 
                 extractive_model: ExtractiveQAModel = None,
                 generative_model: GenerativeQAModel = None,
                 confidence_threshold: float = 0.5):

        self.extractive_model = extractive_model or ExtractiveQAModel()
        self.generative_model = generative_model or GenerativeQAModel()
        self.confidence_threshold = confidence_threshold

        logger.info(f"Hybrid QA initialized with confidence threshold: {confidence_threshold}")

    def answer_question(self, question: str, context: str) -> QAResult:
        """
        Use extractive QA first, fallback to generative if confidence is low
        """
        # Try extractive first
        extractive_result = self.extractive_model.answer_question(question, context)

        # If extractive confidence is high enough, use it
        if extractive_result.confidence >= self.confidence_threshold:
            logger.debug(f"Using extractive answer (confidence: {extractive_result.confidence:.3f})")
            return extractive_result

        # Otherwise, use generative
        generative_result = self.generative_model.answer_question(question, context)

        # Add metadata about the decision
        generative_result.metadata = generative_result.metadata or {}
        generative_result.metadata.update({
            "hybrid_decision": "generative",
            "extractive_confidence": extractive_result.confidence,
            "extractive_answer": extractive_result.answer
        })

        logger.debug(f"Using generative answer (extractive confidence too low: {extractive_result.confidence:.3f})")
        return generative_result

    def get_both_answers(self, question: str, context: str) -> Tuple[QAResult, QAResult]:
        """
        Get both extractive and generative answers for comparison
        """
        extractive_result = self.extractive_model.answer_question(question, context)
        generative_result = self.generative_model.answer_question(question, context)

        return extractive_result, generative_result

# Testing and utility functions
def create_sample_qa_pairs():
    """Create sample question-context pairs for testing"""
    samples = [
        {"question": "Who were the astronauts that landed on the Moon during Apollo 11?"},
        {"question": "Which astronaut did not step on the Moon?"},
        {"question": "Name the individual who stayed behind in orbit while his teammates conducted the lunar landing."},
        {"question": "During which decade did NASA first manage to land humans on the Moon?"},
        {"question": "Why was the Apollo 11 mission controversial despite its success?"},
        {"question": "How did the Apollo missions indirectly influence technological progress on Earth?"},
        {"question": "What was the program's most critical milestone, and when did it happen?"},
        {"question": " What does it refer to in the phrase it was not without controversy?"},
        {"question": "Did every astronaut on Apollo 11 walk on the Moon?"},
        {"question": "Was the Apollo 11 mission considered a failure? Explain briefly."},
        {"question": "Was Yuri Gagarin part of the Apollo 11 mission?"},
        {"question": "Which geopolitical factor motivated the U.S. to pursue the Apollo program?"}
    ]

    return samples

def test_qa_models():
    """Test function for QA models"""
    print("Testing QA Models...")
    print("="*70)
    context="""In the early 1960s, NASA faced significant challenges in its race to land a human on the Moon. The Apollo program required not only technological innovation but also unprecedented coordination among thousands of engineers, scientists, and contractors.

The program's most critical milestone was achieved in July 1969, when Apollo 11 successfully landed astronauts Neil Armstrong and Buzz Aldrin on the lunar surface. Michael Collins, the third astronaut on the mission, remained in lunar orbit aboard the command module.

While the mission was celebrated as a triumph of human ingenuity, it was not without controversy. Many questioned the enormous expenditure, particularly as the U.S. faced domestic issues such as the Vietnam War and civil rights unrest. Nevertheless, the mission established the United States as a global leader in space exploration.

Interestingly, the Apollo missions also contributed to advancements in materials science, computer technology, and telemetry. Many technologies initially developed for lunar exploration later found applications in consumer electronics and industrial automation."""

    # Create test samples
    samples = create_sample_qa_pairs()

    # Test Extractive QA
    print("\n1. Testing Extractive QA (DistilBERT)...")
    print("-"*70)
    try:
        extractive_qa = ExtractiveQAModel()

        for i, sample in enumerate(samples):
            result = extractive_qa.answer_question(sample["question"], context)
            print(f"\nQ{i+1}: {sample['question']}")
            print(f"A{i+1}: {result.answer}")
            print(f"Confidence: {result.confidence:.3f}")

    except Exception as e:
        print(f"❌ Extractive QA failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Generative QA  
    print("\n\n2. Testing Generative QA (T5)...")
    print("-"*70)
    try:
        generative_qa = GenerativeQAModel()

        for i, sample in enumerate(samples):
            result = generative_qa.answer_question(sample["question"], context)
            print(f"\nQ{i+1}: {sample['question']}")
            print(f"A{i+1}: {result.answer}")
            print(f"Confidence: {result.confidence:.3f}")

    except Exception as e:
        print(f"❌ Generative QA failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Hybrid QA
    print("\n\n3. Testing Hybrid QA...")
    print("-"*70)
    try:
        hybrid_qa = HybridQAModel(confidence_threshold=0.6)

        # FIXED: Use first sample instead of entire list
        for i, sample in enumerate(samples):
           
           result = hybrid_qa.answer_question(sample["question"], context)
           print(f"\nQ: {sample['question']}")
           print(f"A: {result.answer}")
           print(f"Model used: {result.model_type}")
           print(f"Confidence: {result.confidence:.3f}")

        # Show both answers for comparison
           ext_result, gen_result = hybrid_qa.get_both_answers(sample["question"], context)
           print(f"\nComparison:")
           print(f"Extractive: {ext_result.answer} (conf: {ext_result.confidence:.3f})")
           print(f"Generative: {gen_result.answer} (conf: {gen_result.confidence:.3f})")

    except Exception as e:
        print(f"❌ Hybrid QA failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("✅ QA model testing completed!")
    print("="*70)

if __name__ == "__main__":

    test_qa_models()
