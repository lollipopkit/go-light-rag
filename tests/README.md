# go-light-rag Benchmark

This directory contains benchmarks that compare two RAG (Retrieval-Augmented Generation) approaches:

1. **LightRAG**: A knowledge graph-augmented RAG system
2. **NaiveRAG**: A traditional vector-based RAG system

The benchmark measures performance across multiple documents and queries, evaluating aspects like response quality, retrieval speed, and token efficiency.

## Setup Instructions

### 1. Configure the Environment

Copy the example configuration file and update it with your credentials:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to include your:
- Neo4J connection details
- LLM provider credentials
- Embedding API key
- Logging preferences

The benchmark supports multiple LLM providers:
- **OpenAI**: Configure with API key and model name
- **Anthropic**: Configure with API key, model name, and max tokens
- **Ollama**: Configure with host URL and model name
- **OpenRouter**: Configure with API key and model name

Example configuration for different providers:

```yaml
# OpenAI Configuration
rag_llm:
  type: "openai"
  api_key: "your-openai-api-key"
  model: "gpt-4o-mini"
  parameters:
    temperature: 0.7

# Anthropic Configuration
rag_llm:
  type: "anthropic"
  api_key: "your-anthropic-api-key"
  model: "claude-3-haiku-20240307"
  max_tokens: 4096
  parameters:
    temperature: 0.7

# Ollama Configuration
rag_llm:
  type: "ollama"
  host: "http://localhost:11434"
  model: "llama3"
  parameters:
    temperature: 0.7
```

### 2. Download Required Documents

The benchmark uses these documents from Project Gutenberg:

1. [A Christmas Carol by Charles Dickens](https://www.gutenberg.org/ebooks/24022)
2. [The Prophet by Kahlil Gibran](https://www.gutenberg.org/ebooks/58585)
3. [Relativity: the Special and General Theory by Albert Einstein](https://www.gutenberg.org/ebooks/5001)

Download these files and place them in a `docs` directory within the `tests` folder:

```bash
mkdir -p docs
cd docs
# Download each document as a .txt file
# Rename files to match: christmascarol.txt, prophet.txt, relativity.txt
```

### 3. Start a Neo4J Instance

You'll need a running Neo4J instance for the benchmark. You can:
- Use a free [Neo4J AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) instance
- Run a local instance with Docker:
  ```bash
  docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest
  ```

### 4. Run the Benchmark

From the `tests` directory, run:

```bash
go test -bench=. -timeout=0
```

The `-timeout=0` flag is important as the benchmark can take significant time due to:
- Document insertion process
- LLM query evaluation
- Multi-document comparison

## Benchmark Results

Below are the results of running this benchmark with the `gpt-4o-mini` model:

```
goos: linux
goarch: amd64
pkg: github.com/lollipopkit/go-light-rag/tests
cpu: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
BenchmarkRAGSystems/christmascarol-8         	       1	158226631759 ns/op	      2706 01_Light_ms/query	      1163 02_Naive_ms/query	       100.0 03_Light_win%	         0 04_Naive_win%	       100.0 05_Light_comp_win%	         0 06_Naive_comp_win%	        50.00 07_Light_div_win%	        50.00 08_Naive_div_win%	       100.0 09_Light_emp_win%	         0 10_Naive_emp_win%	     30974 11_Light_tokens/query	      6332 12_Naive_tokens/query
BenchmarkRAGSystems/relativity-8             	       1	173939083846 ns/op	      3127 01_Light_ms/query	       837.5 02_Naive_ms/query	        83.33 03_Light_win%	        16.67 04_Naive_win%	        83.33 05_Light_comp_win%	        16.67 06_Naive_comp_win%	        33.33 07_Light_div_win%	        66.67 08_Naive_div_win%	        83.33 09_Light_emp_win%	        16.67 10_Naive_emp_win%	     26364 11_Light_tokens/query	      6327 12_Naive_tokens/query
BenchmarkRAGSystems/prophet-8                	       1	136281591193 ns/op	      2705 01_Light_ms/query	       904.5 02_Naive_ms/query	        50.00 03_Light_win%	        50.00 04_Naive_win%	        66.67 05_Light_comp_win%	        33.33 06_Naive_comp_win%	         0 07_Light_div_win%	       100.0 08_Naive_div_win%	        50.00 09_Light_emp_win%	        50.00 10_Naive_emp_win%	     15801 11_Light_tokens/query	      6331 12_Naive_tokens/query
PASS
ok  	github.com/lollipopkit/go-light-rag/tests	468.711s
```

### Understanding the Metrics

Each benchmark result includes several metrics that compare LightRAG and NaiveRAG systems:

| Metric | Description |
|--------|-------------|
| `01_Light_ms/query` | Average query time in milliseconds for LightRAG |
| `02_Naive_ms/query` | Average query time in milliseconds for NaiveRAG |
| `03_Light_win%` | Percentage of queries where LightRAG was the overall winner |
| `04_Naive_win%` | Percentage of queries where NaiveRAG was the overall winner |
| `05_Light_comp_win%` | Percentage of queries where LightRAG won on comprehensiveness |
| `06_Naive_comp_win%` | Percentage of queries where NaiveRAG won on comprehensiveness |
| `07_Light_div_win%` | Percentage of queries where LightRAG won on diversity of perspectives |
| `08_Naive_div_win%` | Percentage of queries where NaiveRAG won on diversity of perspectives |
| `09_Light_emp_win%` | Percentage of queries where LightRAG won on empowerment |
| `10_Naive_emp_win%` | Percentage of queries where NaiveRAG won on empowerment |
| `11_Light_tokens/query` | Average token usage per query for LightRAG |
| `12_Naive_tokens/query` | Average token usage per query for NaiveRAG |

### Summary of Results with gpt-4o-mini

| Document | Query Time (ms) | Overall Win % | Comp Win % | Div Win % | Emp Win % | Token Usage |
|----------|-----------------|---------------|------------|-----------|-----------|-------------|
|          | Light / Naive   | Light / Naive | Light / Naive | Light / Naive | Light / Naive | Light / Naive |
| Christmas Carol | 2706 / 1163 | 100% / 0% | 100% / 0% | 50% / 50% | 100% / 0% | 30974 / 6332 |
| Relativity | 3127 / 837.5 | 83.33% / 16.67% | 83.33% / 16.67% | 33.33% / 66.67% | 83.33% / 16.67% | 26364 / 6327 |
| The Prophet | 2705 / 904.5 | 50% / 50% | 66.67% / 33.33% | 0% / 100% | 50% / 50% | 15801 / 6331 |

### Try Different Configurations

We encourage you to run the benchmark with different configurations to see how various factors affect performance:

- Try different LLM models (claude-3-haiku, llama3, etc.)
- Adjust the temperature and other LLM parameters
- Experiment with different documents and query types
- Modify the chunking size in the handler configuration

Performance may vary significantly depending on your choice of LLM, document complexity, and query types. The benchmark framework makes it easy to test these variations and find the optimal configuration for your specific use case.

## Notes

- This benchmark uses `gpt-4o-mini` as the default LLM model in examples
- First runs will take longer as documents must be processed and indexed
- Results may vary based on the LLM model quality and query complexity
- You can adjust the timeout if needed for your environment
