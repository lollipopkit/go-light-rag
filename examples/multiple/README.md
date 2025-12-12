# go-light-rag Multiple Document Example

This example demonstrates how to use the `go-light-rag` library to build a RAG system that processes multiple documents. The implementation showcases how to handle different file types with specialized handlers and efficiently process document changes across multiple runs.

## Prerequisites

Before running this example, you'll need:

- Either:
  - A graph database (Neo4J or MemGraph) + local files (`vec.db`, `kv.db`), or
  - PostgreSQL with `pgvector` enabled (single-database mode)
- An OpenAI API key
- A directory of text/code files for processing

## Setup Instructions

### 1. Clone & Navigate

After cloning the repository, change to the example directory:

```bash
cd example/multiple
```

### 2. Set Up Graph Database

Choose one of these options:
- Create a free account on [Neo4J](https://neo4j.com/)
- Self-host a compatible graph database like [MemGraph](https://memgraph.com/)

Make note of your graph database URI, username, and password.

**PostgreSQL alternative:** set `postgres_conn_string` in `config.yaml` to use PostgreSQL (with `pgvector`) as the only storage backend. In this mode, Neo4J/ChromeM/Bolt are not used for RAG storage.

### 3. Get OpenAI API Key

Create an account on [OpenAI](https://openai.com/) and generate an API key.

### 4. Prepare Your Documents Directory

Create a directory containing the files you want to analyze. The system supports various file types with specialized handling for Go source files.

### 5. Configure the Application

Copy the example configuration file and update it with your credentials:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to include your:
- Neo4J connection details
- OpenAI API key and model
- Document directory path (`docs_dir`)
- Custom entity types (optional)
- Preferred logging level

## Running the Example

Execute the application:

```bash
go run main.go
```

**First run:** The system will process all files in your documents directory and build the RAG database. This can take significant time depending on the size of your codebase (approximately 1 hour for the entire go-light-rag codebase).

**Subsequent runs:** The system will check for changed files using hash comparisons and only process those that have been modified since the previous run.

## How It Works

The system:
1. Walks through your document directory (respecting `.gitignore` files)
2. Selects appropriate handlers based on file extensions (Go handler for `.go` files, Default handler for others)
3. Stores document hashes to track changes between runs
4. Embeds document text using OpenAI
5. Stores vectors in ChromeM and metadata in BoltDB
6. Creates a knowledge graph in Neo4J
7. Provides an interactive query interface with handler selection
8. Retrieves relevant information for each query using the selected handler
9. Uses a prompt template to ensure consistent output quality

## Key Features

### Multiple Handler Support

The example provides specialized handling for different file types:
- **Default Handler**: For general text documents
- **Go Handler**: Specialized for Go source code files

### Incremental Processing

The system uses file hashing to detect changes:
- Only processes new or modified files on subsequent runs
- Saves significant time when working with large document collections

### GitIgnore Support

The system respects `.gitignore` files in your document directory:
- Automatically skips files that match gitignore patterns
- Follows standard gitignore rules for path matching

### Custom Entity Types

You can configure custom entity types in the configuration file to better suit your document domain:
```yaml
defaultEntityTypes:
  - "config"
  - "dependency"
  - "reference"
  # ... other domain-specific entity types
```

## Query Interface

When running the example, you'll be prompted to:
1. Select a handler type for your query (Default or Go)
2. Enter your query
3. Review the results, which include both vector retrieval data and knowledge graph relationships

This allows you to choose the most appropriate handler for each specific query, improving result quality.

## Demo

[![asciicast](https://asciinema.org/a/709629.svg)](https://asciinema.org/a/709629)

*Note: This demo uses the `go-light-rag` codebase itself as the knowledge source and runs with the `gpt-4o-mini` model. The initial document insertion process (which takes approximately 1 hour) is not included in the demo recording.*
