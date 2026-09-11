# Self-RAG — Self-Reflective Retrieval-Augmented Generation

An advanced **Self-RAG (Self-Reflective Retrieval-Augmented Generation)** system built with **LangGraph, LangChain, FAISS, Hugging Face embeddings, and Groq LLMs**.

The system does not blindly retrieve documents for every query. Instead, it **decides whether retrieval is necessary, evaluates retrieved documents, generates an answer, verifies whether the answer is supported by the retrieved context, automatically revises unsupported answers, evaluates answer usefulness, and rewrites the retrieval query when the answer is not useful.**

This creates a self-reflective RAG workflow capable of iteratively improving its retrieval and generation process.

---

## 🚀 Project Overview

Traditional RAG generally follows:

```text
User Question
     ↓
Retrieve Documents
     ↓
Generate Answer
```

This approach can fail when:

* Retrieval is unnecessary
* Retrieved documents are irrelevant
* The generated answer contains unsupported claims
* The answer does not actually address the question
* The original retrieval query is poor
* The model needs another retrieval attempt

This project implements a **Self-RAG architecture** where the system continuously evaluates its own decisions and outputs.

### Self-RAG Workflow

```text
                    ┌─────────────────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Decide Retrieval Needed? │
                 └────────────┬─────────────┘
                       ┌──────┴──────┐
                       │             │
                     False          True
                       │             │
                       ▼             ▼
                Direct Answer     Retrieve
                                     │
                                     ▼
                            Relevance Evaluation
                                     │
                              ┌──────┴──────┐
                              │             │
                         Relevant       Not Relevant
                              │             │
                              ▼             ▼
                         Generate      No Answer Found
                              │
                              ▼
                         IsSUP Check
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Fully Supported     Unsupported/
                    │              Partially Supported
                    │                   │
                    ▼                   ▼
                 IsUSE              Revise Answer
                    │                   │
             ┌──────┴──────┐            │
             │             │            │
           Useful      Not Useful       │
             │             │            │
             ▼             ▼            │
            END       Rewrite Query ◄───┘
                           │
                           ▼
                        Retrieve
                           │
                           ▼
                      Repeat Cycle
```

---

# ✨ Key Features

## 1. Adaptive Retrieval

The system first determines whether external/internal documents are required.

For example:

```text
Question:
What is NexaAI's refund policy?

→ Retrieval Required
```

While:

```text
Question:
What is machine learning?

→ Direct Answer
```

This prevents unnecessary vector database searches for general questions.

---

## 2. Internal Document Retrieval

The project loads company-specific PDF documents:

```text
documents/
├── Company_Policies.pdf
├── Company_Profile.pdf
└── Product_and_Pricing.pdf
```

The PDFs are split into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
)
```

These chunks are converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stored inside a FAISS vector database.

---

## 3. Semantic Vector Search

The system uses:

```text
Hugging Face Embeddings
        ↓
      FAISS
        ↓
Similarity Search
        ↓
Top 4 Documents
```

Retriever configuration:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)
```

The system retrieves the four most semantically similar chunks for the query.

---

# 🧠 Self-RAG Reflection Pipeline

The main strength of this project is the reflection mechanism.

The system evaluates its own:

* Retrieval decision
* Document relevance
* Answer grounding
* Answer quality/usefulness
* Retrieval query

---

# 1. Retrieval Decision

Before retrieval, an LLM determines whether retrieval is actually required.

### Schema

```python
class RetrieveDecision(BaseModel):
    should_retrieve: bool
```

The model returns:

```json
{
  "should_retrieve": true
}
```

or:

```json
{
  "should_retrieve": false
}
```

### Routing

```text
                    Question
                       │
                       ▼
              Retrieval Decision
                 /           \
                /             \
             True             False
              │                 │
              ▼                 ▼
           Retrieve        Direct Answer
```

---

# 2. Document Relevance Evaluation

After retrieval, every retrieved document is individually evaluated.

The system checks whether the document is relevant to the question.

### Decision Schema

```python
class RelevanceDecision(BaseModel):
    is_relevant: bool
```

The evaluator follows a **topic-level relevance strategy**.

For example:

```text
Question:
What is the refund policy?

Document:
Product and Pricing

→ Relevant
```

While:

```text
Question:
What is the refund policy?

Document:
Company history

→ Not Relevant
```

Only relevant documents are passed to generation.

---

# 3. Context-Based Answer Generation

Once relevant documents are identified, they are combined into a context block.

```text
Relevant Documents
        ↓
Context Construction
        ↓
LLM
        ↓
Generated Answer
```

The model is instructed to answer based on the internal company information.

Example:

```text
Question:
What is NexaAI's refund policy?

Context:
[Relevant company policy information]

Answer:
[Generated answer]
```

---

# 4. IsSUP — Answer Support Verification

One of the most important Self-RAG components is **IsSUP**.

After generating an answer, the system checks:

> Is the generated answer actually supported by the retrieved context?

The verifier classifies the answer into three categories:

```text
fully_supported
partially_supported
no_support
```

### Schema

```python
class IsSUPDecision(BaseModel):
    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]

    evidence: List[str]
```

---

## IsSUP Decision Logic

### Fully Supported

Every meaningful claim in the answer is directly supported by the context.

```text
Context:
Employees receive 20 annual leave days.

Answer:
Employees receive 20 annual leave days.

→ fully_supported
```

---

### Partially Supported

The main facts are supported, but the answer adds unsupported interpretation.

```text
Context:
Employees receive 20 annual leave days.

Answer:
The company offers a generous and employee-friendly
annual leave policy of 20 days.

→ partially_supported
```

The words:

```text
generous
employee-friendly
```

are not explicitly supported by the context.

---

### No Support

The answer contains claims that cannot be supported by the retrieved context.

```text
Context:
Company provides cloud-based analytics products.

Answer:
The company was founded in 2018 by three engineers.

→ no_support
```

If that information does not exist in the context, the answer is unsupported.

---

# 5. Evidence Extraction

The IsSUP evaluator also extracts supporting evidence.

Example:

```text
issup:
fully_supported

evidence:
- "Employees receive 20 annual leave days."
- "Annual leave must be approved by the manager."
```

The project limits evidence to a small number of direct context quotes.

---

# 6. Automatic Answer Revision

If the answer is:

```text
partially_supported
```

or:

```text
no_support
```

the system does not immediately return the answer.

Instead, it enters a **revision loop**.

```text
Generated Answer
       ↓
     IsSUP
       ↓
Unsupported
       ↓
Revise Answer
       ↓
    IsSUP
       ↓
Supported?
```

The revision model is instructed to use only direct quotes from the available context.

Example format:

```text
- "Employees receive 20 annual leave days."
- "Annual leave must be approved by the manager."
```

This reduces hallucination and unsupported interpretation.

---

# 7. Maximum Revision Limit

To prevent infinite loops, the project uses:

```python
MAX_RETRIES = 10
```

The system can perform up to ten answer revision attempts before moving forward.

This is important for production systems because reflection loops must always have a termination condition.

---

# 8. IsUSE — Answer Usefulness Evaluation

Even if an answer is grounded, it may not actually answer the user's question.

For this reason, the project includes a second reflection step:

**IsUSE**

The system evaluates whether the answer is useful for the original question.

### Possible Outputs

```text
useful
not_useful
```

### Example

Question:

```text
What is NexaAI's refund period?
```

Answer:

```text
NexaAI provides several payment options for customers.
```

The answer may be factually related to the company but does not answer the actual question.

Therefore:

```text
isuse = not_useful
```

---

# 9. Retrieval Query Rewriting

If the answer is not useful, the system does not simply stop.

It rewrites the original question into a better vector retrieval query.

Example:

```text
Original Question:
Do NexaAI plans include a free trial?
```

Rewritten retrieval query:

```text
NexaAI free trial duration trial period plans
```

Another example:

```text
Original:
What is NexaAI refund policy?
```

Rewritten:

```text
NexaAI refund policy cancellation refund timeline charges
```

The rewritten query is then sent back to the retriever.

---

# 🔄 Iterative Retrieval Loop

This creates the main Self-RAG feedback loop:

```text
Question
   ↓
Retrieve
   ↓
Relevance Evaluation
   ↓
Generate
   ↓
IsSUP
   ↓
IsUSE
   ↓
Not Useful?
   │
   └────── Yes
             ↓
       Rewrite Retrieval Query
             ↓
          Retrieve Again
             ↓
        Generate Again
```

The maximum number of retrieval-query rewrites is:

```python
MAX_REWRITE_TRIES = 3
```

This prevents uncontrolled retrieval loops.

---

# 🏗️ Architecture

```text
┌───────────────────────────────────────────────────────┐
│                    Self-RAG System                    │
├───────────────────────────────────────────────────────┤
│                                                       │
│  User Question                                       │
│       │                                               │
│       ▼                                               │
│  Retrieval Decision                                   │
│       │                                               │
│   ┌───┴────┐                                          │
│   │        │                                          │
│ Direct   Retrieval                                    │
│   │        │                                          │
│   │        ▼                                          │
│   │   FAISS Retriever                                 │
│   │        │                                          │
│   │        ▼                                          │
│   │   Relevance Filter                               │
│   │        │                                          │
│   │        ▼                                          │
│   │   Context Generation                             │
│   │        │                                          │
│   │        ▼                                          │
│   │      IsSUP                                       │
│   │        │                                          │
│   │   ┌────┴─────┐                                    │
│   │   │          │                                    │
│   │ Supported  Unsupported                            │
│   │   │          │                                    │
│   │   ▼          ▼                                    │
│   │ IsUSE    Revise Answer                            │
│   │   │          │                                    │
│   │   │          └──────► IsSUP                       │
│   │   │                                               │
│   │   ├── Useful ───────► END                        │
│   │   │                                               │
│   │   └── Not Useful                                   │
│   │          │                                        │
│   │          ▼                                        │
│   │    Rewrite Query                                  │
│   │          │                                        │
│   │          └──────────► Retrieve                   │
│   │                                                       │
└───────────────────────────────────────────────────────┘
```

---

# 🔗 LangGraph Workflow

The entire workflow is implemented using **LangGraph StateGraph**.

### Nodes

```text
decide_retrieval
generate_direct
retrieve
is_relevant
generate_from_context
no_answer_found
is_sup
revise_answer
is_use
rewrite_question
```

### Graph

```text
START
  │
  ▼
decide_retrieval
  │
  ├──────────────► generate_direct ─────► END
  │
  ▼
retrieve
  │
  ▼
is_relevant
  │
  ├──────────────► no_answer_found ─────► END
  │
  ▼
generate_from_context
  │
  ▼
is_sup
  │
  ├──────────────► revise_answer
  │                    │
  │                    └──────► is_sup
  │
  ▼
is_use
  │
  ├──────────────► END
  │
  ├──────────────► rewrite_question
  │                    │
  │                    ▼
  │                 retrieve
  │
  └──────────────► no_answer_found ─────► END
```

---

# 🧩 Project Components

| Component             | Technology                     |
| --------------------- | ------------------------------ |
| Programming Language  | Python                         |
| LLM                   | Groq                           |
| LLM Model             | Llama 3.3 70B Versatile        |
| Orchestration         | LangGraph                      |
| LLM Framework         | LangChain                      |
| Embeddings            | Hugging Face                   |
| Embedding Model       | all-MiniLM-L6-v2               |
| Vector Database       | FAISS                          |
| Document Loader       | PyPDFLoader                    |
| Text Splitter         | RecursiveCharacterTextSplitter |
| Structured Output     | Pydantic                       |
| Web Search            | Tavily                         |
| Environment Variables | python-dotenv                  |

---

# 📁 Project Structure

Recommended structure:

```text
advanced-self-rag/
│
├── documents/
│   ├── Company_Policies.pdf
│   ├── Company_Profile.pdf
│   └── Product_and_Pricing.pdf
│
├── self_rag.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/advanced-self-rag.git
cd advanced-self-rag
```

---

## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

If you are installing packages manually:

```powershell
pip install langchain
pip install langgraph
pip install langchain-community
pip install langchain-groq
pip install langchain-huggingface
pip install langchain-text-splitters
pip install faiss-cpu
pip install sentence-transformers
pip install pypdf
pip install tavily-python
pip install python-dotenv
pip install pydantic
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Never commit `.env` to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

# ▶️ Running the Project

Run:

```powershell
python self_rag.py
```

The system will:

```text
1. Receive the user question
2. Decide whether retrieval is required
3. Retrieve relevant documents
4. Evaluate document relevance
5. Generate an answer
6. Verify answer grounding
7. Revise unsupported answers
8. Evaluate answer usefulness
9. Rewrite retrieval query if necessary
10. Retrieve again
11. Return the final answer
```

---

# 🧪 Example

### Input

```text
What is NexaAI Solutions?
```

### Initial Retrieval Query

```text
What is the refund policy of NexaAI
```

If the retrieved documents are not useful, the system can automatically generate a better retrieval query based on the original question and previous answer.

Example:

```text
NexaAI Solutions company profile organization overview
```

The system then performs another retrieval cycle.

---

# 📊 Debugging and Inspection

The project includes execution-level inspection output.

Example:

```text
===== RAG EXECUTION RESULT =====

Question: What is NexaAI Solutions?

Need Retrieval: True

Rewrite tries (retrieval): 1
Support revise tries: 0

Retrieval:
  Total retrieved docs: 4
  Relevant docs: 3

Verification (IsSUP):
  issup: fully_supported

  evidence:
   - Company profile evidence...
   - Company description evidence...

Usefulness (IsUSE):
  isuse: useful
  reason: The answer directly explains what NexaAI Solutions is.

Final Answer:
NexaAI Solutions is ...

===============================
```

This makes it easier to inspect the internal decision-making process.

---

# 🎯 Why Self-RAG?

Traditional RAG assumes:

```text
Retrieved documents = Good
Generated answer = Correct
```

Self-RAG challenges these assumptions.

Instead, it asks:

```text
Do I need retrieval?
        ↓
Are these documents relevant?
        ↓
Is my answer supported?
        ↓
Does my answer actually answer the question?
        ↓
If not, can I improve retrieval?
```

This makes the system more reliable and adaptive than a basic RAG pipeline.

---

# 🆚 Traditional RAG vs Self-RAG

| Feature               | Traditional RAG |     Self-RAG |
| --------------------- | --------------: | -----------: |
| Retrieval             |   Always/mostly |     Adaptive |
| Document Evaluation   | Usually limited |          Yes |
| Relevance Filtering   |        Optional |          Yes |
| Answer Grounding      |  Usually absent |        IsSUP |
| Evidence Extraction   |  Usually absent |          Yes |
| Answer Revision       |              No |          Yes |
| Usefulness Evaluation |              No |        IsUSE |
| Query Rewriting       |        Optional |          Yes |
| Iterative Retrieval   |      Usually no |          Yes |
| Reflection            |         Limited | Core feature |
| LangGraph Workflow    |        Optional |          Yes |

---

# 🧠 Core Self-RAG Concepts Demonstrated

This project demonstrates several important RAG and Self-RAG concepts:

### Retrieval

```text
Semantic Search
Vector Embeddings
FAISS
Top-K Retrieval
```

### Reflection

```text
Retrieval Decision
Document Relevance
Answer Support
Answer Usefulness
```

### Generation

```text
Context-Aware Generation
Grounded Answers
Direct Generation
```

### Correction

```text
Answer Revision
Query Rewriting
Iterative Retrieval
```

### Control Flow

```text
Conditional Routing
Feedback Loops
Retry Limits
State Management
```

---

# 🛡️ Hallucination Reduction

The system uses multiple mechanisms to reduce hallucination:

### 1. Relevant Document Filtering

Irrelevant documents are removed before generation.

### 2. Context-Based Generation

The RAG generation stage uses retrieved company documents.

### 3. IsSUP Verification

The generated answer is checked against the context.

### 4. Answer Revision

Unsupported answers are automatically revised.

### 5. Evidence Extraction

The verifier identifies supporting context.

### 6. Usefulness Evaluation

The final answer is checked to determine whether it actually addresses the user's question.

---

# 🔁 Retry & Loop Protection

Self-reflective systems can potentially enter infinite loops.

This implementation uses explicit limits:

```python
MAX_RETRIES = 10
```

for answer revision and:

```python
MAX_REWRITE_TRIES = 3
```

for retrieval-query rewriting.

The LangGraph invocation also uses:

```python
config={"recursion_limit": 80}
```

This provides additional protection against uncontrolled graph execution.

---

# 📈 Future Improvements

The current implementation is a strong Self-RAG prototype. It can be extended toward production with:

## Retrieval Improvements

* Hybrid search
* BM25 + vector search
* Cross-encoder reranking
* Metadata filtering
* Multi-query retrieval
* Parent-document retrieval
* Query decomposition
* Multi-vector retrieval

## RAG Improvements

* Context compression
* Sentence-level filtering
* Context deduplication
* Citation generation
* Source attribution
* Confidence scoring

## Evaluation

* RAGAS
* DeepEval
* LangSmith
* Retrieval evaluation
* Faithfulness evaluation
* Answer relevance
* Context relevance
* Regression testing

## Production Infrastructure

* Qdrant / Pinecone / Weaviate
* PostgreSQL
* Redis
* FastAPI
* Docker
* CI/CD
* AWS / GCP / Azure
* Authentication
* Rate limiting
* Observability
* Monitoring
* Structured logging

## LLMOps

* Prompt versioning
* Model versioning
* Experiment tracking
* Token/cost monitoring
* Latency monitoring
* Automated evaluation
* Production tracing

---

# 🏭 Potential Production Architecture

A future production version could follow:

```text
                    ┌───────────────┐
                    │   React UI    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   LangGraph   │
                    │   Self-RAG    │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        Vector DB         LLM          Redis
        Qdrant/etc.      Groq/etc.     Cache
             │
             ▼
       Document Storage
       S3 / Database
```

---

# 💡 Use Cases

This Self-RAG architecture can be adapted for:

* Enterprise knowledge assistants
* Company policy chatbots
* HR assistants
* Product documentation assistants
* Customer support
* Internal knowledge bases
* Pricing assistants
* Legal document assistants
* Financial document QA
* Technical documentation search
* Research assistants

---

# 🔬 What Makes This Project Different?

This is not simply:

```text
PDF → Embeddings → FAISS → LLM
```

It implements a reflective decision-making pipeline:

```text
                ┌───────────────────┐
                │   User Question   │
                └─────────┬─────────┘
                          │
                          ▼
                  Need Retrieval?
                          │
                          ▼
                    Retrieve Docs
                          │
                          ▼
                  Are Docs Relevant?
                          │
                          ▼
                    Generate Answer
                          │
                          ▼
                  Is Answer Supported?
                       /        \
                     Yes         No
                      │           │
                      ▼           ▼
                  Is Useful?   Revise Answer
                   /     \          │
                 Yes      No        │
                  │        │        │
                  ▼        ▼        │
                 END   Rewrite Query│
                           │        │
                           └────────┘
```

This feedback-driven architecture is the core idea behind **Self-RAG**.

---

# 🧰 Technologies Used

```text
Python
LangChain
LangGraph
Groq
Llama 3.3 70B
Hugging Face
Sentence Transformers
FAISS
Pydantic
PyPDF
Tavily
python-dotenv
```

---

# 📚 Learning Outcomes

By studying this project, you can understand how to build:

* Adaptive RAG systems
* Self-reflective AI pipelines
* LangGraph state machines
* Conditional graph routing
* Retrieval evaluation
* Grounded generation
* Hallucination detection
* Answer revision loops
* Query rewriting
* Iterative retrieval
* Structured LLM outputs
* RAG evaluation workflows
* Production-oriented GenAI architectures

---

# 👨‍💻 Author

**Shayan Ahmed**

AI/ML Engineer | Generative AI | LLMs | RAG | Agentic AI | Full-Stack Development

GitHub: `Mrshayan07`

---

# 📄 License

This project is available under the MIT License.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

It helps support further development of advanced **RAG, Self-RAG, Agentic AI, and LLMOps** projects.
