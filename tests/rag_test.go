package tests

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"text/template"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/internal"
	"github.com/lollipopkit/go-light-rag/llm"
	"github.com/lollipopkit/go-light-rag/storage"
	"github.com/cespare/xxhash"
	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	bolt "go.etcd.io/bbolt"
	"gopkg.in/yaml.v2"
)

// EvaluationResult represents the structured output from the LLM evaluation.
type EvaluationResult struct {
	Comprehensiveness struct {
		Winner      string `json:"winner"`
		Explanation string `json:"explanation"`
	} `json:"comprehensiveness"`
	Diversity struct {
		Winner      string `json:"winner"`
		Explanation string `json:"explanation"`
	} `json:"diversity"`
	Empowerment struct {
		Winner      string `json:"winner"`
		Explanation string `json:"explanation"`
	} `json:"empowerment"`
	OverallWinner struct {
		Winner      string `json:"winner"`
		Explanation string `json:"explanation"`
	} `json:"overall_winner"`
}

type documentMetrics struct {
	TotalQueries   int
	LightWins      int
	NaiveWins      int
	LightQueryTime time.Duration
	NaiveQueryTime time.Duration
	EvalTime       time.Duration
	CompWins       map[string]int // Comprehensiveness wins by system
	DiversityWins  map[string]int // Diversity wins by system
	EmpowerWins    map[string]int // Empowerment wins by system
	LightTokens    int            // Total tokens used by lightRAG prompts
	NaiveTokens    int            // Total tokens used by naiveRAG prompts
}

type config struct {
	Neo4JURI      string `yaml:"neo4j_uri"`
	Neo4JUser     string `yaml:"neo4j_user"`
	Neo4JPassword string `yaml:"neo4j_password"`

	RAGLLM          llmConfig `yaml:"rag_llm"`
	EvalLLM         llmConfig `yaml:"eval_llm"`
	EmbeddingAPIKey string    `yaml:"embedding_api_key"` // For embedding

	LogLevel string `yaml:"log_level"`
}

type llmConfig struct {
	Type       string         `yaml:"type"` // openai, anthropic, ollama, openrouter
	APIKey     string         `yaml:"api_key"`
	Model      string         `yaml:"model"`
	Host       string         `yaml:"host"`       // for Ollama
	MaxTokens  int            `yaml:"max_tokens"` // for Anthropic
	Parameters llm.Parameters `yaml:"parameters"`
}

type storageWrapper struct {
	storage.Bolt
	storage.Chromem
	storage.Neo4J
}

type lightRAG struct {
	llm     golightrag.LLM
	storage storageWrapper
	logger  *slog.Logger
}

type naiveRAG struct {
	sourcesColl *chromem.Collection
	kvDB        storage.Bolt
	logger      *slog.Logger
}

type naivePromptData struct {
	History string
	Chunks  string
}

type lightPromptData struct {
	History     string
	QueryResult string
}

type evalPromptData struct {
	Question string
	Answer1  string
	Answer2  string
}

const (
	configPath = "config.yaml"
	hashBucket = "hash"

	naiveColl = "naive"
)

//nolint:lll
const naivePrompt = `---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{{.History}}

---Document Chunks---
{{.Chunks}}

---Response Rules---

- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] Source content (File: file_path)
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks.`

//nolint:lll
const lightPrompt = `---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{{.History}}

---Knowledge Base---
{{.QueryResult}}

---Response Rules---

- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] Source content (File: file_path)
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base.`

//nolint:lll
const evalPrompt = `---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{{.Question}}

Here are the two answers:

**Answer 1:**
{{.Answer1}}

**Answer 2:**
{{.Answer2}}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

---Output Format---
Return ONLY a JSON object with no additional text or formatting. Your entire response must be valid JSON in exactly this format:
{
    "comprehensiveness": {
        "winner": "[Answer 1 or Answer 2]",
        "explanation": "[Provide explanation here]"
    },
    "diversity": {
        "winner": "[Answer 1 or Answer 2]",
        "explanation": "[Provide explanation here]"
    },
    "empowerment": {
        "winner": "[Answer 1 or Answer 2]",
        "explanation": "[Provide explanation here]"
    },
    "overall_winner": {
        "winner": "[Answer 1 or Answer 2]",
        "explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }
}`

var documents = []string{
	"christmascarol.txt",
	"relativity.txt",
	"prophet.txt",
}

//nolint:lll
var entitiesTypes = map[string][]string{
	"christmascarol.txt": {"character", "organization", "location", "time period", "object", "theme", "event"},
	"relativity.txt":     {"scientific concept", "physical phenomenon", "mathematical formula", "thought experiment", "scientific principle", "reference frame", "physical quantity", "scientific instrument", "scientist", "theory"},
	"prophet.txt":        {"philosophical concept", "emotion", "life stage", "relationship", "virtue", "human experience", "metaphor", "spiritual element", "natural element", "abstract quality"},
}

//nolint:lll
var queriesMap = map[string][]string{
	"christmascarol.txt": {
		"How do the supernatural visitors influence Scrooge's relationships with the Cratchit family throughout different time periods shown in the narrative?",
		"What connections exist between the locations visited during the ghostly journeys and the thematic lessons about wealth and compassion?",
		"Analyze how objects like the chains carried by Marley relate to the themes of redemption and personal transformation for characters in Victorian London.",
		"Examine the relationships between poverty in Camden Town and the attitudes of businessmen at the Exchange, particularly regarding charity during the winter season.",
		"How do events from Scrooge's past at Fezziwig's establishment connect to his present relationships with Bob Cratchit and his nephew Fred?",
		"Compare Tiny Tim's influence on the Cratchit household with the Ghost of Christmas Yet to Come's influence on Scrooge's transformation.",
	},
	"relativity.txt": {
		"How do reference frames and physical quantities interact differently in Newtonian mechanics compared to the relativistic framework?",
		"Analyze the relationship between the constancy of light speed and the transformation of space-time measurements observed from different inertial systems.",
		"What connections exist between thought experiments involving moving trains and the mathematical formulations of simultaneity in different reference frames?",
		"Examine how the principle of equivalence relates gravitational fields to accelerated reference frames and their effects on physical phenomena.",
		"How do the concepts of curved spacetime geometry interact with planetary orbits and light paths near massive objects?",
		"Compare the relationships between energy, mass, and momentum in different physical scenarios where relativistic effects become significant.",
	},
	"prophet.txt": {
		"How do Almustafa's teachings about love and marriage relate to his perspectives on children and giving across the various chapters of wisdom?",
		"Analyze the connections between the symbolic imagery of the sea and Almustafa's message about freedom and self-knowledge throughout his discourses.",
		"What relationships exist between Almitra's role as a seeress and the metaphorical concepts of time and eternity discussed in the philosophical dialogues?",
		"Examine how the setting of Orphalese influences the reception of spiritual wisdom and the contrast between communal and individual understanding.",
		"How do the poetic descriptions of nature's elements mirror the philosophical concepts of joy and sorrow expressed in the teachings?",
		"Compare the metaphorical significance of departure and return with the spiritual journey described in discussions of self-discovery and enlightenment.",
	},
}

func BenchmarkRAGSystems(b *testing.B) {
	// Common setup - done once for all benchmarks
	cfg, ragLLM, evalLLM, logger, defaultHandler, err := setupBenchmarkEnvironment()
	if err != nil {
		b.Fatalf("Failed to set up benchmark environment: %v", err)
	}

	// Create RAG systems - done once
	lRAG, err := newLightRAG(cfg, ragLLM, logger)
	if err != nil {
		b.Fatalf("Error creating lightRAG: %v\n", err)
	}
	defer func() {
		if err := lRAG.close(); err != nil {
			b.Errorf("Error closing lightRAG: %v\n", err)
		}
	}()

	nRAG, err := newNaiveRAG(cfg.EmbeddingAPIKey, logger)
	if err != nil {
		b.Fatalf("Error creating naiveRAG: %v\n", err)
	}

	// Run benchmarks for each document
	for _, doc := range documents {
		docName := strings.TrimSuffix(doc, ".txt")
		b.Run(docName, func(b *testing.B) {
			// Ensure documents are inserted before benchmarking
			docPath := "docs/" + doc
			defaultHandler.EntityTypes = entitiesTypes[doc]

			if err := lRAG.insert(docPath, defaultHandler); err != nil {
				b.Fatalf("Error inserting document %s to lightRAG: %v", doc, err)
			}

			if err := nRAG.insert(docPath, defaultHandler.ChunksDocument); err != nil {
				b.Fatalf("Error inserting document %s to naiveRAG: %v", doc, err)
			}

			// Run actual benchmark
			for b.Loop() {
				benchRAGSystem(b, doc, ragLLM, evalLLM, lRAG, nRAG, defaultHandler, logger)
			}
		})
	}
}

func setupBenchmarkEnvironment() (*config, golightrag.LLM, golightrag.LLM, *slog.Logger, handler.Default, error) {
	// Load configuration from YAML file
	cfg, err := loadConfig(configPath)
	if err != nil {
		return nil, nil, nil, nil, handler.Default{}, fmt.Errorf("error loading configuration: %w", err)
	}

	// Set log level based on configuration
	logLevel := slog.LevelInfo
	switch strings.ToLower(cfg.LogLevel) {
	case "debug":
		logLevel = slog.LevelDebug
	case "warn":
		logLevel = slog.LevelWarn
	case "error":
		logLevel = slog.LevelError
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	}))

	// Set up RAG LLM
	ragLLM, err := buildLLM(cfg.RAGLLM, logger)
	if err != nil {
		return nil, nil, nil, nil, handler.Default{}, fmt.Errorf("error creating RAG LLM: %w", err)
	}

	// Set up Evaluation LLM
	evalLLM, err := buildLLM(cfg.EvalLLM, logger)
	if err != nil {
		return nil, nil, nil, nil, handler.Default{}, fmt.Errorf("error creating Evaluation LLM: %w", err)
	}

	defaultHandler := handler.Default{
		ChunkMaxTokenSize: 1200,
		Config: handler.DocumentConfig{
			MaxRetries:       5,
			BackoffDuration:  3 * time.Second,
			ConcurrencyCount: 5,
		},
	}

	return cfg, ragLLM, evalLLM, logger, defaultHandler, nil
}

func benchRAGSystem(
	b *testing.B,
	documentName string,
	ragLLM golightrag.LLM,
	evalLLM golightrag.LLM,
	lRAG lightRAG,
	nRAG naiveRAG,
	defaultHandler handler.Default,
	logger *slog.Logger,
) {
	// Define metrics struct
	metrics := &documentMetrics{
		CompWins:      make(map[string]int),
		DiversityWins: make(map[string]int),
		EmpowerWins:   make(map[string]int),
	}

	// Reset the timer to exclude setup costs
	b.ResetTimer()

	// Run benchmark for this document
	queries := queriesMap[documentName]
	defaultHandler.EntityTypes = entitiesTypes[documentName]

	for _, query := range queries {
		history := fmt.Sprintf("role: user, content: %s", query)

		// Measure lightRAG
		logger.Info("lightRAG query", "document", documentName, "query", query)
		lightResult, lightDuration, err := lRAG.query(query, defaultHandler)
		if err != nil {
			b.Errorf("lightRAG error on query %s: %v", query, err)
			continue
		}
		metrics.LightQueryTime += lightDuration

		lPromptData := lightPromptData{
			History:     history,
			QueryResult: lightResult,
		}
		buf := strings.Builder{}
		tmpl := template.New("light-prompt")
		tmpl = template.Must(tmpl.Parse(lightPrompt))
		if err := tmpl.Execute(&buf, lPromptData); err != nil {
			b.Errorf("Error executing template: %v", err)
			continue
		}
		lightPromptText := buf.String()
		lightCountToken, err := internal.CountTokens(lightPromptText)
		if err != nil {
			b.Errorf("Error counting tokens on light: %v", err)
			continue
		}
		metrics.LightTokens += lightCountToken
		lightAnswer, err := ragLLM.Chat([]string{lightPromptText})
		if err != nil {
			b.Errorf("Error calling LLM on light: %v", err)
			continue
		}
		logger.Debug("LLM response for light", "response", lightAnswer)

		// Measure naiveRAG
		logger.Info("naiveRAG query", "document", documentName, "query", query)
		naiveResult, naiveDuration, err := nRAG.query(query)
		if err != nil {
			b.Errorf("naiveRAG error on query %s: %v", query, err)
			continue
		}
		metrics.NaiveQueryTime += naiveDuration

		nPromptData := naivePromptData{
			History: history,
			Chunks:  naiveResult,
		}
		buf = strings.Builder{}
		tmpl = template.New("naive-prompt")
		tmpl = template.Must(tmpl.Parse(naivePrompt))
		if err := tmpl.Execute(&buf, nPromptData); err != nil {
			b.Errorf("Error executing template: %v\n", err)
			continue
		}
		naivePromptText := buf.String()
		naiveCountToken, err := internal.CountTokens(naivePromptText)
		if err != nil {
			b.Errorf("Error counting tokens on naive: %v", err)
			continue
		}
		metrics.NaiveTokens += naiveCountToken
		naiveAnswer, err := ragLLM.Chat([]string{naivePromptText})
		if err != nil {
			b.Errorf("Error calling LLM on naive: %v\n", err)
			continue
		}
		logger.Debug("LLM response for naive", "response", naiveAnswer)

		// LLM Evaluation
		ePromptData := evalPromptData{
			Question: query,
			Answer1:  lightAnswer,
			Answer2:  naiveAnswer,
		}
		buf = strings.Builder{}
		tmpl = template.New("eval-prompt")
		tmpl = template.Must(tmpl.Parse(evalPrompt))
		if err := tmpl.Execute(&buf, ePromptData); err != nil {
			b.Errorf("Error executing template: %v", err)
			continue
		}
		evalPromptText := buf.String()
		start := time.Now()
		evalResult, err := evalLLM.Chat([]string{evalPromptText})
		if err != nil {
			b.Errorf("LLM evaluation error on query %s: %v", query, err)
			continue
		}
		evalDuration := time.Since(start)
		metrics.EvalTime += evalDuration

		logger.Info("LLM evaluation result", "document", documentName, "result", evalResult)

		// Parse results
		var result EvaluationResult
		if err := json.Unmarshal([]byte(evalResult), &result); err != nil {
			b.Errorf("Failed to parse LLM response for query %s: %v", query, err)
			continue
		}

		// Update metrics
		metrics.TotalQueries++

		// Track categorical wins
		metrics.CompWins[result.Comprehensiveness.Winner]++
		metrics.DiversityWins[result.Diversity.Winner]++
		metrics.EmpowerWins[result.Empowerment.Winner]++

		// Track overall winner
		switch result.OverallWinner.Winner {
		case "Answer 1":
			metrics.LightWins++
		case "Answer 2":
			metrics.NaiveWins++
		}
	}

	// Report document metrics
	if metrics.TotalQueries > 0 {
		// Timing metrics
		b.ReportMetric(float64(metrics.LightQueryTime)/float64(metrics.TotalQueries)/float64(time.Millisecond),
			"01_Light_ms/query")
		b.ReportMetric(float64(metrics.NaiveQueryTime)/float64(metrics.TotalQueries)/float64(time.Millisecond),
			"02_Naive_ms/query")

		// Win percentages
		b.ReportMetric(float64(metrics.LightWins)/float64(metrics.TotalQueries)*100,
			"03_Light_win%")
		b.ReportMetric(float64(metrics.NaiveWins)/float64(metrics.TotalQueries)*100,
			"04_Naive_win%")

		// Comprehensiveness metrics
		b.ReportMetric(float64(metrics.CompWins["Answer 1"])/float64(metrics.TotalQueries)*100,
			"05_Light_comp_win%")
		b.ReportMetric(float64(metrics.CompWins["Answer 2"])/float64(metrics.TotalQueries)*100,
			"06_Naive_comp_win%")

		// Diversity metrics
		b.ReportMetric(float64(metrics.DiversityWins["Answer 1"])/float64(metrics.TotalQueries)*100,
			"07_Light_div_win%")
		b.ReportMetric(float64(metrics.DiversityWins["Answer 2"])/float64(metrics.TotalQueries)*100,
			"08_Naive_div_win%")

		// Empowerment metrics
		b.ReportMetric(float64(metrics.EmpowerWins["Answer 1"])/float64(metrics.TotalQueries)*100,
			"09_Light_emp_win%")
		b.ReportMetric(float64(metrics.EmpowerWins["Answer 2"])/float64(metrics.TotalQueries)*100,
			"10_Naive_emp_win%")

		// Count Tokens metrics
		b.ReportMetric(float64(metrics.LightTokens)/float64(metrics.TotalQueries),
			"11_Light_tokens/query")
		b.ReportMetric(float64(metrics.NaiveTokens)/float64(metrics.TotalQueries),
			"12_Naive_tokens/query")
	}
}

func loadConfig(path string) (*config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %w", err)
	}

	var cfg config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("error parsing config file: %w", err)
	}

	return &cfg, nil
}

func buildLLM(cfg llmConfig, logger *slog.Logger) (golightrag.LLM, error) {
	switch strings.ToLower(cfg.Type) {
	case "openai":
		return llm.NewOpenAI(cfg.APIKey, cfg.Model, cfg.Parameters, logger), nil
	case "anthropic":
		return llm.NewAnthropic(cfg.APIKey, cfg.Model, cfg.MaxTokens, cfg.Parameters), nil
	case "ollama":
		return llm.NewOllama(cfg.Host, cfg.Model, cfg.Parameters, logger), nil
	case "openrouter":
		return llm.NewOpenRouter(cfg.APIKey, cfg.Model, cfg.Parameters, logger), nil
	default:
		return nil, fmt.Errorf("unsupported LLM type: %s", cfg.Type)
	}
}

func createHashBucket(kvDB storage.Bolt) error {
	return kvDB.DB.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(hashBucket))
		return err
	})
}

func checkFileHash(kvDB storage.Bolt, fileID, content string) (bool, error) {
	var hash uint64

	err := kvDB.DB.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(hashBucket))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		hashBs := b.Get([]byte(fileID))
		if len(hashBs) == 0 {
			hash = 0
			return nil
		}

		hash = binary.BigEndian.Uint64(hashBs)
		return nil
	})
	if err != nil {
		return false, fmt.Errorf("error checking file hash: %w", err)
	}

	contentHash := xxhash.Sum64String(content)

	return hash != contentHash, nil
}

func saveFileHash(kvDB storage.Bolt, fileID, content string) error {
	return kvDB.DB.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(hashBucket))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		hash := xxhash.Sum64String(content)
		hashBs := binary.BigEndian.AppendUint64(nil, hash)

		return b.Put([]byte(fileID), hashBs)
	})
}

func newLightRAG(cfg *config, llm golightrag.LLM, logger *slog.Logger) (lightRAG, error) {
	graphDB, err := storage.NewNeo4J(cfg.Neo4JURI, cfg.Neo4JUser, cfg.Neo4JPassword)
	if err != nil {
		return lightRAG{}, fmt.Errorf("error creating neo4jDB: %w", err)
	}

	vecDB, err := storage.NewChromem("vec.db", 5,
		storage.EmbeddingFunc(chromem.NewEmbeddingFuncOpenAI(cfg.EmbeddingAPIKey, chromem.EmbeddingModelOpenAI3Large)))
	if err != nil {
		return lightRAG{}, fmt.Errorf("error creating chromemDB: %w", err)
	}

	kvDB, err := storage.NewBolt("kv.db")
	if err != nil {
		return lightRAG{}, fmt.Errorf("error creating boltDB: %w", err)
	}
	if err := createHashBucket(kvDB); err != nil {
		return lightRAG{}, fmt.Errorf("error creating hash bucket: %w", err)
	}

	store := storageWrapper{
		Bolt:    kvDB,
		Chromem: vecDB,
		Neo4J:   graphDB,
	}

	return lightRAG{
		llm:     llm,
		storage: store,
		logger:  logger.With(slog.String("rag", "light")),
	}, nil
}

func newNaiveRAG(openAIAPIKey string, logger *slog.Logger) (naiveRAG, error) {
	vecDB, err := chromem.NewPersistentDB("naive_vec.db", false)
	if err != nil {
		return naiveRAG{}, fmt.Errorf("error creating chromemDB: %w", err)
	}

	coll, err := vecDB.GetOrCreateCollection(naiveColl, nil,
		chromem.NewEmbeddingFuncOpenAI(openAIAPIKey, chromem.EmbeddingModelOpenAI3Large))
	if err != nil {
		return naiveRAG{}, fmt.Errorf("error creating chromem collection: %w", err)
	}

	kvDB, err := storage.NewBolt("naive_kv.db")
	if err != nil {
		return naiveRAG{}, fmt.Errorf("error creating boltDB: %w", err)
	}
	if err := createHashBucket(kvDB); err != nil {
		return naiveRAG{}, fmt.Errorf("error creating hash bucket: %w", err)
	}

	return naiveRAG{
		sourcesColl: coll,
		kvDB:        kvDB,
		logger:      logger.With(slog.String("rag", "naive")),
	}, nil
}

func (l lightRAG) insert(path string, handler golightrag.DocumentHandler) error {
	// Read file content
	fileData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	fileContent := string(fileData)

	// Generate a file ID based on the path
	fileID := strings.ReplaceAll(path, string(filepath.Separator), "_")

	// Check if file has changed by comparing hash
	shouldInsert, err := checkFileHash(l.storage.Bolt, fileID, fileContent)
	if err != nil {
		return fmt.Errorf("error checking file hash: %w", err)
	}

	if !shouldInsert {
		l.logger.Debug("File unchanged, skipping", "path", path)
		return nil
	}

	l.logger.Info("Inserting file", "path", path)

	// Insert document
	doc := golightrag.Document{
		ID:      fileID,
		Content: fileContent,
	}

	now := time.Now()
	if err := golightrag.Insert(doc, handler, l.storage, l.llm, l.logger); err != nil {
		return fmt.Errorf("error inserting document: %w", err)
	}

	l.logger.Info("Inserted document", "id", doc.ID, "duration in milliseconds", time.Since(now).Milliseconds())

	// Save new hash
	if err := saveFileHash(l.storage.Bolt, fileID, fileContent); err != nil {
		return fmt.Errorf("error saving file hash: %w", err)
	}

	return nil
}

func (l lightRAG) query(query string, handler golightrag.QueryHandler) (string, time.Duration, error) {
	start := time.Now()

	answer, err := golightrag.Query([]golightrag.QueryConversation{
		{
			Role:    golightrag.RoleUser,
			Message: query,
		},
	}, handler, l.storage, l.llm, l.logger)
	if err != nil {
		return "", 0, fmt.Errorf("error querying: %w", err)
	}

	return answer.String(), time.Since(start), nil
}

func (l lightRAG) close() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	if err := l.storage.Close(ctx); err != nil {
		return fmt.Errorf("error closing neo4jDB: %w", err)
	}

	return nil
}

func (n naiveRAG) insert(path string, chunkingFunc func(string) ([]golightrag.Source, error)) error {
	// Read file content
	fileData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	fileContent := string(fileData)

	// Generate a file ID based on the path
	fileID := strings.ReplaceAll(path, string(filepath.Separator), "_")

	// Check if file has changed by comparing hash
	shouldInsert, err := checkFileHash(n.kvDB, fileID, fileContent)
	if err != nil {
		return fmt.Errorf("error checking file hash: %w", err)
	}

	if !shouldInsert {
		n.logger.Debug("File unchanged, skipping", "path", path)
		return nil
	}

	chunks, err := chunkingFunc(fileContent)
	if err != nil {
		return fmt.Errorf("error chunking file: %w", err)
	}

	n.logger.Info("Inserting file", "path", path, "count chunks", len(chunks))

	for _, chunk := range chunks {
		doc := chromem.Document{
			ID:      uuid.New().String(),
			Content: chunk.Content,
		}

		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()

		if err := n.sourcesColl.AddDocument(ctx, doc); err != nil {
			return fmt.Errorf("error adding document to collection: %w", err)
		}
	}

	// Save new hash
	if err := saveFileHash(n.kvDB, fileID, fileContent); err != nil {
		return fmt.Errorf("error saving file hash: %w", err)
	}

	return nil
}

func (n naiveRAG) query(query string) (string, time.Duration, error) {
	start := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vecRes, err := n.sourcesColl.Query(ctx, query, 5, nil, nil)
	if err != nil {
		return "", 0, fmt.Errorf("error querying: %w", err)
	}

	res := make([]string, len(vecRes))
	for i, vec := range vecRes {
		res[i] = vec.Content
	}

	resStr := strings.Join(res, `
##########
`)

	return resStr, time.Since(start), nil
}
