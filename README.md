🤖 Enhanced Hybrid RAG Question Answering Agent

A production-style Retrieval-Augmented Generation (RAG) system that combines local document retrieval, Wikipedia-based external retrieval, and hybrid QA models to answer questions with traceable sources, confidence scores, and evaluation metrics.

The system supports:

1. User-uploaded documents (PDF, DOCX, TXT)

2. Hybrid retrieval (FAISS + Wikipedia)

3. Extractive, Generative, and Hybrid QA

4.  Quantitative evaluation (F1, Exact Match)

5.  Interactive Streamlit UI

6.  Batch inference & health monitoring
=============================================================================================================================================================
System Architecture
User Question
      ↓
Hybrid Retriever
 ├─ FAISS Vector Retriever (local + uploaded docs)
 └─ Wikipedia Retriever (current / external info)
      ↓
Context Aggregation & Truncation
      ↓
QA Model
 ├─ Extractive (DistilBERT)
 ├─ Generative (T5)
 └─ Hybrid (confidence-based switching)
      ↓
Answer + Confidence + Sources
=============================================================================================================================================================
.
├── enhanced_app.py              # Streamlit UI (chat, uploads, batch, health)
├── rag_pipeline.py              # Core RAG orchestration logic
├── inference.py                 # Unified inference API + statistics
├── enhanced_retriever.py        # FAISS + document upload + Wikipedia retriever
├── qa_model.py                  # Extractive, Generative, Hybrid QA models
├── evaluate.py                  # F1 / Exact Match evaluation pipeline
├── test_data.json               # QA evaluation dataset
├── evaluation_results.json      # Stored evaluation outputs
├── download_squad_dataset.py    # Dataset helper (SQuAD subset)
├── requirements_enhanced.txt    # Dependencies
└── README.md
=============================================================================================================================================================Retrieval Layer
##Enhanced Vector Retriever

Sentence embeddings via SentenceTransformers

FAISS cosine similarity (IP + L2 normalization)

Automatic chunking with overlap

Persistent index (faiss_index.bin)

Metadata-aware chunks (source, upload date, file type)

Supported file formats:.txt,.pdf,.docx
##Enhanced Wikipedia Retriever

Live Wikipedia querying

Query rewriting for current events

Disambiguation handling

Source URLs + retrieval timestamps

##Hybrid Retrieval Strategy

Results from multiple retrievers are merged and ranked, enabling:

Closed-book knowledge (uploaded docs)

Open-world knowledge (Wikipedia)
============================================================================================================================================================
Question Answering Models
-Extractive QA (Default)

DistilBERT (SQuAD-finetuned)

Span-based answer extraction

Token-level confidence estimation

-Generative QA

T5-based generation

Used when extractive confidence is low

Heuristic + log-probability confidence scoring

-Hybrid QA (Recommended)

Runs extractive QA first

Falls back to generative QA if confidence < threshold

Preserves both answers in metadata
============================================================================================================================================================
Evaluation Framework

The system includes quantitative evaluation using:

Exact Match (EM)

Token-level F1 score

Average latency per question

Example Results (500 QA samples)
Model	F1 Score	                                   Exact Match	                                Avg Time
Extractive (DistilBERT)                           	0.781	64.0%	                                  83 ms
Generative (T5)	                                    0.758	61.4%	                                201 ms
Hybrid QA                                         	0.787	65.4%	                                  144 ms

Hybrid QA achieves the best accuracy–latency tradeoff.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hell yeahhhhhhhhhhh
##Quick Usage (Programmatic)
from inference import get_api_instance

api = get_api_instance()
response = api.ask("What is machine learning?")

print(response["answer"])
print(response["confidence"])

============================================================================================================================================================
--Limitations

Wikipedia dependency introduces latency

Generative QA may paraphrase beyond sources

No reranker / cross-encoder yet

No multilingual support (currently English-only)

--Future Improvements

Cross-encoder reranking

Long-context LLM integration

Multi-document citation grouping

Active learning from low-confidence queries

Multilingual retrieval & QA
