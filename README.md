# Enterprise Knowledge Graph Connector

An AI-powered enterprise knowledge management system that combines Retrieval-Augmented Generation (RAG), knowledge graphs, and hybrid retrieval to answer questions from uploaded enterprise documents with supporting evidence.

## Features

- **Document ingestion:** Upload PDF, DOCX, and TXT documents.
- **Document management:** List, update, and delete uploaded documents.
- **Text preprocessing:** Clean and normalize extracted document content.
- **Chunking and embeddings:** Split documents into chunks and generate vector embeddings.
- **Hybrid retrieval:** Combine ChromaDB vector retrieval with Neo4j knowledge graph information.
- **Entity extraction:** Identify people, projects, departments, and other enterprise entities.
- **Entity resolution:** Resolve aliases and merge references to the same entity.
- **Relationship extraction:** Identify relationships such as `WORKS_ON`, `MANAGES`, `BELONGS_TO`, and `OWNS`.
- **Graph reasoning:** Support multi-hop reasoning over connected entities.
- **Grounded question answering:** Generate answers using retrieved evidence.
- **Knowledge graph visualization:** Explore entities and relationships through the Streamlit interface.
- **Evaluation:** Evaluate retrieval, extraction, grounding, and application workflows.

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Language | Python |
| Language processing | spaCy |
| Vector database | ChromaDB |
| Graph database | Neo4j |
| Embedding model | sentence-transformers |
| LLM | Google Gemini |
| Testing | pytest |

## System Architecture

The system follows a hybrid Graph RAG architecture.

1. Users upload enterprise documents through the frontend.
2. The backend extracts, cleans, and chunks document content.
3. Text chunks are embedded and stored in ChromaDB.
4. Entity and relationship information is extracted and stored in Neo4j.
5. User questions are processed through hybrid retrieval.
6. Retrieved vector and graph evidence is used to generate grounded answers.
7. The frontend displays answers, document information, and knowledge graph visualizations.

## Project Structure

```text
Enterprise-Knowledge-Graph-Connector/
│
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── ingestion/
├── retrieval/
├── extraction/
├── knowledge_graph/
├── vector_store/
├── evaluation/
├── tests/
├── README.md
└── requirements.txt
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/gayathriravikumar10a-dotcom/Enterprise-Knowledge-Graph-Connector.git
cd Enterprise-Knowledge-Graph-Connector
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Configure the required Gemini API key and Neo4j connection settings in your local environment according to the application's configuration.

Do not commit API keys, passwords, or other secrets to GitHub.

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 6. Start the frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/upload` | Upload a document |
| GET | `/documents` | List uploaded documents |
| PUT | `/documents/{document_id}` | Update a document |
| DELETE | `/documents/{document_id}` | Delete a document |
| POST | `/ask` | Ask a question |
| GET | `/graph` | Retrieve knowledge graph data |

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Evaluation

### Retrieval Evaluation

The preliminary retrieval evaluation dataset contains three queries across two documents.

| Metric | Result |
|---|---:|
| Precision@5 | 0.40 |
| Recall@5 | 1.00 |
| MRR | 1.00 |
| Average retrieval latency | 141.61 ms |

**Metric interpretation:**
- Precision@5 measures the proportion of relevant chunks among the top five retrieved chunks.
- Recall@5 measures whether relevant documents were retrieved in the top five.
- MRR measures the ranking position of the first relevant result.
- Retrieval latency measures the time taken for retrieval.

These results are preliminary and should not be interpreted as performance across a large enterprise dataset.

### Automated Testing

Latest recorded full test-suite result:

**107 tests passed, 2 warnings.**

Additional workflow validation:

**3 workflow tests passed.**

The project includes tests covering ingestion, document processing, retrieval, entity extraction, entity resolution, relationship extraction, graph reasoning, visualization, grounding, and application workflows.

## Limitations

- Retrieval evaluation currently uses a small dataset and should be expanded.
- The grounding evaluator uses lexical overlap heuristics and does not independently establish semantic entailment.
- Answer quality depends on document quality, retrieval performance, and LLM availability.
- Production deployment requires appropriate database configuration, authentication, monitoring, and secret management.

## Contributors

This project is developed collaboratively, with responsibilities covering core GenAI development and integration, document processing, knowledge graph and NLP development, and testing and deployment readiness.

## License

Add the project's selected license before distributing or reusing the repository.

