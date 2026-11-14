\# Enterprise Knowledge Graph Connector



An AI-powered enterprise document intelligence system that combines \*\*Retrieval-Augmented Generation (RAG), Knowledge Graphs, Vector Search, and Large Language Models\*\* to retrieve connected information from enterprise documents and generate evidence-grounded answers.



\## Project Overview



Enterprise organizations store valuable information across PDFs, Word documents, and text files. Traditional keyword-based search often struggles to understand relationships between employees, departments, projects, and business processes.



The Enterprise Knowledge Graph Connector addresses this challenge through a hybrid Graph RAG architecture that combines semantic document retrieval with knowledge graph relationships.



Users can upload documents, ask natural-language questions, and receive answers supported by retrieved document and graph evidence.



\## Key Features



\- \*\*Multi-format document ingestion:\*\* Supports PDF, DOCX, and TXT files.

\- \*\*Text preprocessing and chunking:\*\* Splits documents into manageable chunks with metadata.

\- \*\*Named Entity Recognition:\*\* Extracts entities using spaCy.

\- \*\*Relationship extraction:\*\* Identifies relationships using rule-based patterns and dependency parsing.

\- \*\*Knowledge graph construction:\*\* Represents entities and relationships using NetworkX.

\- \*\*Graph database storage:\*\* Stores knowledge graph data in Neo4j.

\- \*\*Vector database:\*\* Stores document embeddings in ChromaDB.

\- \*\*Semantic search:\*\* Uses Sentence Transformers for embeddings and cosine similarity for retrieval.

\- \*\*Hybrid retrieval:\*\* Combines graph-based evidence with vector search results.

\- \*\*Grounded answer generation:\*\* Uses Gemini to generate answers based on retrieved evidence.

\- \*\*Grounding evaluation:\*\* Checks whether generated answers are supported by retrieved evidence using a lexical-overlap heuristic.

\- \*\*REST API:\*\* Provides document upload and question-answering endpoints using FastAPI.

\- \*\*Interactive frontend:\*\* Provides document upload and question-answering functionality through Streamlit.



\## Technology Stack



| Component | Technology |

|---|---|

| Programming language | Python |

| NLP and entity extraction | spaCy |

| Relationship extraction | Regex and dependency parsing |

| Graph representation | NetworkX |

| Graph database | Neo4j |

| Embedding model | Sentence Transformers (`all-MiniLM-L6-v2`) |

| Vector database | ChromaDB |

| Similarity search | Cosine similarity |

| LLM | Google Gemini |

| RAG architecture | Hybrid Graph RAG |

| Backend | FastAPI |

| Frontend | Streamlit |

| Testing | Pytest |



\## System Architecture



```text

&#x20;                USER

&#x20;                  |

&#x20;                  v

&#x20;          STREAMLIT FRONTEND

&#x20;                  |

&#x20;                  v

&#x20;            FASTAPI BACKEND

&#x20;                  |

&#x20;         +--------+--------+

&#x20;         |                 |

&#x20;         v                 v

&#x20;   DOCUMENT UPLOAD     USER QUESTION

&#x20;         |                 |

&#x20;         v                 v

&#x20;   DOCUMENT LOADER    HYBRID RETRIEVER

&#x20;         |                 |

&#x20;         v           +-----+------+

&#x20;   TEXT CHUNKING      |            |

&#x20;         |            v            v

&#x20;         v       VECTOR SEARCH  GRAPH SEARCH

&#x20;   ENTITY AND          |            |

&#x20;   RELATIONSHIP        v            v

&#x20;   EXTRACTION       CHROMADB      NEO4J

&#x20;         |            |            |

&#x20;         v            +-----+------+

&#x20;   KNOWLEDGE GRAPH          |

&#x20;         |                  v

&#x20;         v             RETRIEVED

&#x20;       NEO4J             EVIDENCE

&#x20;                            |

&#x20;                            v

&#x20;                      GEMINI LLM

&#x20;                            |

&#x20;                            v

&#x20;                    GROUNDED ANSWER

&#x20;                            |

&#x20;                            v

&#x20;                   GROUNDING CHECK

&#x20;                            |

&#x20;                            v

&#x20;                         USER

```



\## Project Workflow



\### 1. Document Ingestion



The system accepts PDF, DOCX, and TXT documents and extracts their text and metadata.



\### 2. Text Preprocessing



Extracted text is cleaned and divided into smaller overlapping chunks for efficient retrieval.



\### 3. Entity Extraction



spaCy identifies relevant entities such as people, projects, and departments.



\### 4. Relationship Extraction



Rule-based patterns and dependency parsing identify relationships such as:



\- WORKS\_ON

\- MANAGES

\- BELONGS\_TO



\### 5. Knowledge Graph Construction



Extracted entities and relationships are represented as a graph using NetworkX and persisted in Neo4j.



\### 6. Vector Indexing



Sentence Transformers generates embeddings for document chunks, which are stored in ChromaDB.



\### 7. Hybrid Retrieval



The retriever combines:

\- Semantic similarity search from ChromaDB.

\- Relationship-based graph retrieval from Neo4j.



\### 8. Answer Generation



Retrieved evidence is passed to Gemini, which generates an answer constrained by the available context.



\### 9. Grounding Evaluation



A lexical-overlap-based evaluator checks whether the answer is supported by retrieved evidence. The current evaluator is a heuristic and does not establish full semantic entailment.



\## Installation and Setup



\### Prerequisites



\- Python 3.11

\- Git

\- Neo4j Desktop or a reachable Neo4j instance

\- Google Gemini API key



\### 1. Clone the repository



```bash

git clone https://github.com/gayathriravikumar10a-dotcom/Enterprise-Knowledge-Graph-Connector.git



cd Enterprise-Knowledge-Graph-Connector

```



\### 2. Create a virtual environment



```powershell

python -m venv venv



.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Download the spaCy model



```bash

python -m spacy download en\_core\_web\_sm

```



\### 5. Configure environment variables



Create a `.env` file in the project root using `.env.example` as a template.



Configure your own credentials:



```env

GEMINI\_API\_KEY=your\_gemini\_api\_key

GEMINI\_MODEL=gemini-3.5-flash



NEO4J\_URI=bolt://localhost:7687

NEO4J\_USERNAME=neo4j

NEO4J\_PASSWORD=your\_neo4j\_password

```



Never commit your actual `.env` file or API credentials to GitHub.



\### 6. Start Neo4j



Start your Neo4j database and verify that the connection settings match your `.env` configuration.



\## Running the Application



Open two PowerShell terminals from the project root.



\### Terminal 1 — Start FastAPI



```powershell

.\\venv\\Scripts\\Activate.ps1



uvicorn backend.main:app --reload

```



API documentation:



http://127.0.0.1:8000/docs



\### Terminal 2 — Start Streamlit



```powershell

.\\venv\\Scripts\\Activate.ps1



streamlit run frontend/app.py

```



Open the local Streamlit URL displayed in your terminal.



\## API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | `/` | Root endpoint |

| GET | `/health` | Health check |

| POST | `/upload` | Upload and process a document |

| POST | `/ask` | Ask a question using hybrid retrieval and RAG |



Interactive API documentation is available at `/docs` when the backend is running.



\## Example Use Case



\*\*Sample document information:\*\*



\- Ravi manages Project Atlas.

\- Asha works on Project Atlas.

\- Project Atlas is an enterprise AI knowledge management initiative.



\*\*Example question:\*\*



Who manages Project Atlas?



\*\*Expected answer:\*\*



Ravi manages Project Atlas.



The answer is generated using retrieved document and graph evidence.



\## Testing



Run the test suite from the project root:



```powershell

pytest -v

```



Tests cover document loading, chunking, entity extraction, relationship extraction, graph construction, reasoning, retrieval, and RAG pipeline behavior.



\## Current Limitations and Future Enhancements



\- Improve entity extraction for previously unseen enterprise entities.

\- Add entity resolution and duplicate detection.

\- Strengthen relationship extraction beyond predefined patterns.

\- Introduce claim-level semantic grounding evaluation.

\- Add document listing, deletion, and lifecycle management.

\- Improve graph visualization and multi-hop reasoning.

\- Expand integration and performance testing.

\- Add authentication, authorization, and production deployment support.



\## Project Structure



```text

Enterprise-Knowledge-Graph-Connector/

│

├── backend/

│   └── main.py

│

├── data/

│   └── sample\_documents/

│

├── evaluation/

│   └── grounding\_evaluator.py

│

├── extraction/

│   ├── entity\_extractor.py

│   └── relationship\_extractor.py

│

├── frontend/

│   └── app.py

│

├── ingestion/

│   ├── document\_loader.py

│   ├── chunker.py

│   └── pipeline.py

│

├── knowledge\_graph/

│   ├── graph\_builder.py

│   ├── neo4j\_store.py

│   ├── reasoning\_engine.py

│   └── workflow.py

│

├── rag/

│   ├── answer\_generator.py

│   └── rag\_pipeline.py

│

├── retrieval/

│   └── hybrid\_retriever.py

│

├── vector\_store/

│   └── chroma\_store.py

│

├── scripts/

├── tests/

├── .env.example

├── .gitignore

├── requirements.txt

└── README.md

```



\## Contributors



Developed as a collaborative AI and Data Science project.



\*\*Core areas:\*\* Generative AI, Retrieval-Augmented Generation, Knowledge Graphs, NLP, Vector Databases, and Backend Development.



\## License



License to be added.

