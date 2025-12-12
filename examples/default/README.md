# go-light-rag Example Implementation

This example demonstrates how to use the `go-light-rag` library to build a document retrieval system. The implementation showcases the library's core functionality in a straightforward manner, using the default handler which closely mirrors the behavior of the official Python LightRAG implementation.

## Prerequisites

Before running this example, you'll need:

- Either:
  - A graph database (Neo4J or MemGraph) + local files (`vec.db`, `kv.db`), or
  - PostgreSQL with `pgvector` enabled (single-database mode)
- An OpenAI API key
- A text document for processing

## Setup Instructions

### 1. Clone & Navigate

After cloning the repository, change to the example directory:

```bash
cd example/default
```

### 2. Set Up Graph Database

Choose one of these options:
- Create a free account on [Neo4J](https://neo4j.com/)
- Self-host a compatible graph database like [MemGraph](https://memgraph.com/)

Make note of your graph database URI, username, and password.

**PostgreSQL alternative:** set `postgres_conn_string` in `config.yaml` to use PostgreSQL (with `pgvector`) as the only storage backend. In this mode, Neo4J/ChromeM/Bolt are not used for RAG storage.

### 3. Get OpenAI API Key

Create an account on [OpenAI](https://openai.com/) and generate an API key.

### 4. Prepare Your Document

Create a text file named `book.txt` containing the document you want to analyze. The default entity types in this example are configured for [A Christmas Carol by Charles Dickens](https://www.gutenberg.org/ebooks/24022).

### 5. Configure the Application

Copy the example configuration file and update it with your credentials:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to include your:
- Neo4J connection details
- OpenAI API key and model
- Preferred logging level

## Running the Example

Execute the application:

```bash
go run main.go
```

**First run:** The system will process your document and build the RAG database (approximately 5 minutes depending on document size).

**Subsequent runs:** As long as the document hasn't changed and the database files (`vec.db` and `kv.db`) exist, the system will skip processing and directly open the query interface.

## How It Works

The system:
1. Embeds document text using OpenAI
2. Stores vectors in ChromeM and metadata in BoltDB
3. Creates a knowledge graph in Neo4J
4. Provides an interactive query interface
5. Retrieves relevant information for each query
6. Uses a prompt template taken directly from the official Python implementation to ensure consistent output quality

## Demo

[![asciicast](https://asciinema.org/a/709605.svg)](https://asciinema.org/a/709605)

*Note: The demo video is played at 2x speed and uses the `gpt-4o-mini` model.*
