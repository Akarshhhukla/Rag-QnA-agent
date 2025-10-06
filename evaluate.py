# evaluate.py
import re
from collections import Counter

def normalize_answer(s):
    """Normalize answer text for comparison"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(predicted, ground_truth):
    """Calculate token-level F1 score between predicted and ground truth answers"""
    pred_tokens = normalize_answer(predicted).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty cases
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0 if len(pred_tokens) != len(truth_tokens) else 1.0
    
    # Count common tokens
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(predicted, ground_truth):
    """Check if predicted answer exactly matches ground truth (after normalization)"""
    return int(normalize_answer(predicted) == normalize_answer(ground_truth))


# # Quick test
# if __name__ == "__main__":
#     # Test the functions
#     pred = "machine learning algorithms"
#     truth = "machine learning"
    
#     f1 = compute_f1(pred, truth)
#     em = compute_exact_match(pred, truth)
    
#     print(f"Predicted: {pred}")
#     print(f"Ground Truth: {truth}")
#     print(f"F1 Score: {f1:.3f}")
#     print(f"Exact Match: {em}")
# Add to evaluate.py

import json
from qa_model import ExtractiveQAModel, GenerativeQAModel, HybridQAModel
import time

def load_test_data(filepath='test_data.json'):
    """Load test dataset from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_model(model, test_data, model_name="Model"):
    """
    Evaluate a QA model on test dataset
    
    Args:
        model: QA model instance (ExtractiveQAModel, GenerativeQAModel, or HybridQAModel)
        test_data: List of test examples
        model_name: Name for display
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    f1_scores = []
    em_scores = []
    processing_times = []
    
    total = len(test_data)
    
    for i, item in enumerate(test_data):
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Progress: {i+1}/{total} questions...")
        
        try:
            # Time the prediction
            start_time = time.time()
            result = model.answer_question(item["question"], item["context"])
            elapsed = time.time() - start_time
            processing_times.append(elapsed)
            
            # Calculate metrics
            f1 = compute_f1(result.answer, item["ground_truth"])
            em = compute_exact_match(result.answer, item["ground_truth"])
            
            f1_scores.append(f1)
            em_scores.append(em)
            
        except Exception as e:
            print(f"  Error on question {i+1}: {e}")
            f1_scores.append(0.0)
            em_scores.append(0)
            processing_times.append(0.0)
    
    # Calculate averages
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    avg_time = sum(processing_times) / len(processing_times)
    
    results = {
        "model_name": model_name,
        "f1_score": avg_f1,
        "exact_match": avg_em,
        "avg_processing_time": avg_time,
        "total_questions": total
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"F1 Score:           {avg_f1:.3f}")
    print(f"Exact Match:        {avg_em:.3f} ({avg_em*100:.1f}%)")
    print(f"Avg Processing:     {avg_time*1000:.1f} ms/question")
    print(f"Total Questions:    {total}")
    print(f"{'='*60}\n")
    
    return results


def run_full_evaluation():
    """Run evaluation on all models"""
    print("Loading test data...")
    test_data = load_test_data('test_data.json')
    print(f"✓ Loaded {len(test_data)} test questions\n")
    
    all_results = {}
    
    # Evaluate Extractive Model
    print("Initializing Extractive QA Model (DistilBERT)...")
    extractive_model = ExtractiveQAModel()
    all_results['extractive'] = evaluate_model(
        extractive_model, 
        test_data, 
        "Extractive QA (DistilBERT)"
    )
    
    # Evaluate Generative Model
    print("\nInitializing Generative QA Model (T5)...")
    generative_model = GenerativeQAModel()
    all_results['generative'] = evaluate_model(
        generative_model, 
        test_data, 
        "Generative QA (T5)"
    )
    
    # Evaluate Hybrid Model
    print("\nInitializing Hybrid QA Model...")
    hybrid_model = HybridQAModel(confidence_threshold=0.5)
    all_results['hybrid'] = evaluate_model(
        hybrid_model, 
        test_data, 
        "Hybrid QA (DistilBERT + T5)"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'F1 Score':<12} {'Exact Match':<12} {'Avg Time (ms)'}")
    print("-"*60)
    
    for key, res in all_results.items():
        print(f"{res['model_name']:<30} {res['f1_score']:<12.3f} {res['exact_match']:<12.3f} {res['avg_processing_time']*1000:<.1f}")
    
    print("="*60)
    
    # Calculate improvement
    if 'extractive' in all_results and 'hybrid' in all_results:
        baseline_f1 = all_results['extractive']['f1_score']
        hybrid_f1 = all_results['hybrid']['f1_score']
        improvement = ((hybrid_f1 - baseline_f1) / baseline_f1) * 100
        
        print(f"\n📊 Hybrid Model Improvement: {improvement:+.1f}% over Extractive baseline")
    
    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to evaluation_results.json")
    
    return all_results


if __name__ == "__main__":
    # Run full evaluation when script is executed
    results = run_full_evaluation()
