# download_squad_subset.py
import json
import requests

def download_squad_subset(num_samples=50):
    """Download a subset of SQuAD dev set"""
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    
    print("Downloading SQuAD dataset...")
    response = requests.get(url)
    squad_data = response.json()
    
    test_data = []
    count = 0
    
    for article in squad_data['data']:
        if count >= num_samples:
            break
            
        for paragraph in article['paragraphs']:
            if count >= num_samples:
                break
                
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                if count >= num_samples:
                    break
                
                # Skip unanswerable questions
                if qa['is_impossible']:
                    continue
                
                question = qa['question']
                answer = qa['answers'][0]['text'] if qa['answers'] else ""
                
                test_data.append({
                    "question": question,
                    "context": context,
                    "ground_truth": answer
                })
                
                count += 1
    
    # Save to file
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Downloaded {len(test_data)} questions")
    print(f"✓ Saved to test_data.json")

if __name__ == "__main__":
    download_squad_subset(500)  # Start with 50, increase to 100+ later
