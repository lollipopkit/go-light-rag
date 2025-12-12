package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strings"
	"text/template"
	"time"

	"github.com/cespare/xxhash"
	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/llm"
	"github.com/lollipopkit/go-light-rag/storage"
	"github.com/philippgille/chromem-go"
	bolt "go.etcd.io/bbolt"
	"gopkg.in/yaml.v2"
)

type config struct {
	Neo4JURI      string `yaml:"neo4j_uri"`
	Neo4JUser     string `yaml:"neo4j_user"`
	Neo4JPassword string `yaml:"neo4j_password"`

	OpenAIAPIKey string `yaml:"openai_api_key"`
	OpenAIModel  string `yaml:"openai_model"`

	PostgresConnString string `yaml:"postgres_conn_string"`
	VectorDim          int    `yaml:"vector_dim"`
	TopK               int    `yaml:"top_k"`

	LogLevel string `yaml:"log_level"`
}

type storageWrapper struct {
	storage.Bolt
	storage.Chromem
	storage.Neo4J
}

type ragPromptData struct {
	History     string
	QueryResult string
}

const (
	docPath    = "book.txt"
	configPath = "config.yaml"
)

//nolint:lll
var ragPrompt = `
---Role---

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

func main() {
	log.SetFlags(0)

	// Load configuration from YAML file
	cfg, err := loadConfig(configPath)
	if err != nil {
		log.Printf("Error loading configuration: %v\n", err)
		return
	}

	defaultHandler := handler.Default{
		ChunkMaxTokenSize: 1500,
		EntityTypes: []string{
			"character", "organization", "location", "time period", "object", "theme", "event",
		},
		Config: handler.DocumentConfig{
			MaxRetries:       5,
			BackoffDuration:  3 * time.Second,
			ConcurrencyCount: 5,
		},
	}

	usePostgres := strings.TrimSpace(cfg.PostgresConnString) != ""

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

	temp := float32(0.1)

	openAI := llm.NewOpenAI(cfg.OpenAIAPIKey, cfg.OpenAIModel, llm.Parameters{
		Temperature: &temp,
	}, logger)

	var (
		store  golightrag.Storage
		hashDB storage.Bolt
	)

	if usePostgres {
		// Use PostgreSQL (pgvector) as the ONLY storage backend.
		vectorDim := cfg.VectorDim
		if vectorDim <= 0 {
			vectorDim = 3072
		}
		topK := cfg.TopK
		if topK <= 0 {
			topK = 5
		}

		pgStore, err := storage.NewPostgres(
			cfg.PostgresConnString,
			vectorDim,
			topK,
			storage.EmbeddingFunc(chromem.NewEmbeddingFuncOpenAI(cfg.OpenAIAPIKey, chromem.EmbeddingModelOpenAI3Large)),
		)
		if err != nil {
			log.Printf("Error creating postgresDB: %v\n", err)
			return
		}
		defer pgStore.DB.Close()

		hashDB, err = storage.NewBolt("hash.db")
		if err != nil {
			log.Printf("Error creating hash boltDB: %v\n", err)
			return
		}

		store = pgStore
	} else {
		graphDB, err := storage.NewNeo4J(cfg.Neo4JURI, cfg.Neo4JUser, cfg.Neo4JPassword)
		if err != nil {
			log.Printf("Error creating neo4jDB: %v\n", err)
			return
		}
		defer func() {
			closeCtx, closeCancel := context.WithTimeout(context.Background(), time.Second*30)
			defer closeCancel()

			if err := graphDB.Close(closeCtx); err != nil {
				log.Printf("Error closing neo4jDB: %v\n", err)
			}
		}()

		vecDB, err := storage.NewChromem("vec.db", 5,
			storage.EmbeddingFunc(chromem.NewEmbeddingFuncOpenAI(cfg.OpenAIAPIKey, chromem.EmbeddingModelOpenAI3Large)))
		if err != nil {
			log.Printf("Error creating chromemDB: %v\n", err)
			return
		}

		kvDB, err := storage.NewBolt("kv.db")
		if err != nil {
			log.Printf("Error creating boltDB: %v\n", err)
			return
		}

		store = storageWrapper{
			Bolt:    kvDB,
			Chromem: vecDB,
			Neo4J:   graphDB,
		}
		hashDB = kvDB
	}

	fileData, err := os.ReadFile(docPath)
	if err != nil {
		log.Printf("Error reading file: %v\n", err)
		return
	}
	docContent := string(fileData)

	// Check the hash of the knowledge base that already inserted into the storage,
	// to determine whether to insert the document or not.
	noInsert, err := checkKGHash(hashDB, docContent)
	if err != nil {
		log.Printf("Error checking knowledge base hash: %v\n", err)
		return
	}

	if !noInsert {
		log.Printf("The document is not in the knowledge base. Inserting...\n")
		if err := insert(docContent, defaultHandler, store, openAI, logger); err != nil {
			log.Printf("Error inserting document: %v\n", err)
			return
		}
		if err := saveKGHash(hashDB, docContent); err != nil {
			log.Printf("Error saving knowledge base hash: %v\n", err)
			return
		}
	}

	// Start the query loop
	query(defaultHandler, store, openAI, logger)
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

func checkKGHash(kvDB storage.Bolt, docContent string) (bool, error) {
	// Use different bucket to store the hash of the document.
	if err := kvDB.DB.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte("hash"))
		return err
	}); err != nil {
		return false, fmt.Errorf("failed to create hash bucket: %w", err)
	}

	var hash uint64

	err := kvDB.DB.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("hash"))
		if b == nil {
			// This should never happen, but just in case.
			return fmt.Errorf("bucket not found")
		}
		hashBs := b.Get([]byte("book"))
		if len(hashBs) == 0 {
			hash = 0
			return nil
		}

		hash = binary.BigEndian.Uint64(hashBs)
		return nil
	})
	if err != nil {
		return false, fmt.Errorf("error checking knowledge base existence: %w", err)
	}

	docHash := xxhash.Sum64String(docContent)

	return hash == docHash, nil
}

func saveKGHash(kvDB storage.Bolt, docContent string) error {
	return kvDB.DB.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("hash"))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		hash := xxhash.Sum64String(docContent)
		hashBs := binary.BigEndian.AppendUint64(nil, hash)

		return b.Put([]byte("book"), hashBs)
	})
}

func insert(
	docContent string,
	docHandler golightrag.DocumentHandler,
	storage golightrag.Storage,
	llm golightrag.LLM,
	logger *slog.Logger,
) error {
	now := time.Now()
	defer func() {
		logger.Info("Inserted document", "duration in milliseconds", time.Since(now).Milliseconds())
	}()

	doc := golightrag.Document{
		ID:      "book",
		Content: docContent,
	}

	return golightrag.Insert(doc, docHandler, storage, llm, logger)
}

func query(handler golightrag.QueryHandler, store golightrag.Storage, llm golightrag.LLM, logger *slog.Logger) {
	// Track conversation for the RAG system
	convo := make([]golightrag.QueryConversation, 0)

	// Maximum turns to keep in conversation history
	const maxTurns = 10

	for {
		log.Println("Insert query: (type 'exit' to exit)")
		reader := bufio.NewReader(os.Stdin)
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Error reading input: %v\n", err)
			return
		}
		line = strings.TrimSpace(line)

		if line == "exit" {
			log.Println("Exiting...")
			return
		}

		logger.Info("User query", "query", line)

		now := time.Now()

		// Add user query to conversation
		convo = append(convo, golightrag.QueryConversation{
			Role:    golightrag.RoleUser,
			Message: line,
		})

		// Keep conversation history within limit
		if len(convo)/2 > maxTurns {
			// Remove oldest turn (user+assistant pair)
			convo = convo[2:]
		}

		// Query the RAG system
		res, err := golightrag.Query(convo, handler, store, llm, logger)
		if err != nil {
			log.Printf("Error querying: %v\n", err)
			return
		}

		logger.Debug("Query result", "result", res)

		logger.Info("Calling LLM", "query duration in milliseconds", time.Since(now).Milliseconds())

		// Format conversation history for the prompt
		convoStr := make([]string, len(convo))
		for i, conv := range convo {
			convoStr[i] = conv.String()
		}

		// Prepare the RAG prompt data
		promptData := ragPromptData{
			History:     strings.Join(convoStr, "\n"),
			QueryResult: res.String(),
		}

		// Format the prompt using the template
		buf := strings.Builder{}
		tmpl := template.New("rag-prompt")
		tmpl = template.Must(tmpl.Parse(ragPrompt))
		if err := tmpl.Execute(&buf, promptData); err != nil {
			log.Printf("Error executing template: %v\n", err)
			return
		}
		promptText := buf.String()

		logger.Debug("Prompt text", "prompt", promptText)

		// Call the LLM with the prepared prompt
		llmResponse, err := llm.Chat([]string{promptText})
		if err != nil {
			log.Printf("Error calling LLM: %v\n", err)
			return
		}

		// Display the LLM response
		log.Println("\nAssistant:")
		log.Println(llmResponse)
		log.Println()

		// Add LLM response to conversation for next turn
		convo = append(convo, golightrag.QueryConversation{
			Role:    golightrag.RoleAssistant,
			Message: llmResponse,
		})
	}
}
