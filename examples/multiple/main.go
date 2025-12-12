package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/cespare/xxhash"
	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/llm"
	"github.com/lollipopkit/go-light-rag/storage"
	"github.com/philippgille/chromem-go"
	ignore "github.com/sabhiram/go-gitignore"
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
	DocsDir  string `yaml:"docs_dir"`

	DefaultEntityTypes []string `yaml:"defaultEntityTypes"`
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
	configPath = "config.yaml"
	hashBucket = "hash"
)

//nolint:lll
const ragPrompt = `
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

	if cfg.DocsDir == "" {
		log.Printf("Document directory not specified in config\n")
		return
	}

	// Default entity types to use if not specified in config
	defaultTypes := []string{
		"character", "organization", "location", "time period", "object", "theme", "event",
	}

	// Use config entity types if available, otherwise use defaults
	entityTypes := defaultTypes
	if len(cfg.DefaultEntityTypes) > 0 {
		entityTypes = cfg.DefaultEntityTypes
	}

	defaultHandler := handler.Default{
		ChunkMaxTokenSize: 1500,
		EntityTypes:       entityTypes,
		Config: handler.DocumentConfig{
			MaxRetries:       5,
			BackoffDuration:  3 * time.Second,
			ConcurrencyCount: 2,
		},
	}

	goHandler := handler.Go{
		Default: defaultHandler,
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

	if err := createHashBucket(hashDB); err != nil {
		log.Printf("Error creating hash bucket: %v\n", err)
		return
	}

	// Process all files in the directory
	if err := processDocumentDirectory(cfg.DocsDir, hashDB, store, defaultHandler, goHandler, openAI, logger); err != nil {
		log.Printf("Error processing document directory: %v\n", err)
		return
	}

	// Start the query loop
	query(defaultHandler, goHandler, store, openAI, logger)
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

func createHashBucket(kvDB storage.Bolt) error {
	return kvDB.DB.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(hashBucket))
		return err
	})
}

//nolint:gocognit,funlen // File processing function with necessary conditional logic
func processDocumentDirectory(
	docDir string,
	kvDB storage.Bolt,
	store golightrag.Storage,
	defaultHandler handler.Default,
	goHandler handler.Go,
	llm golightrag.LLM,
	logger *slog.Logger,
) error {
	// Ensure the root directory path is absolute and clean
	docDir, err := filepath.Abs(docDir)
	if err != nil {
		return fmt.Errorf("error getting absolute path: %w", err)
	}
	docDir = filepath.Clean(docDir)

	// Map to store gitignore matchers by directory
	gitignoreMatchers := make(map[string]*ignore.GitIgnore)

	// First pass: collect all .gitignore files and compile matchers
	err = filepath.Walk(docDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			// Skip .git directory
			if filepath.Base(path) == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		if filepath.Base(path) == ".gitignore" {
			dir := filepath.Dir(path)

			// Compile .gitignore file
			matcher, err := ignore.CompileIgnoreFile(path)
			if err != nil {
				return fmt.Errorf("error compiling .gitignore at %s: %w", path, err)
			}

			gitignoreMatchers[dir] = matcher
			logger.Debug("Compiled .gitignore", "path", path)
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("error walking directory for .gitignore files: %w", err)
	}

	// Second pass: find all files excluding those matched by .gitignore patterns
	var files []string
	err = filepath.Walk(docDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			// Skip .git directory
			if filepath.Base(path) == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip .gitignore files themselves
		if filepath.Base(path) == ".gitignore" {
			return nil
		}

		// Check if file should be ignored
		if shouldIgnoreWithMatchers(path, docDir, gitignoreMatchers) {
			relPath, _ := filepath.Rel(docDir, path)
			logger.Debug("Ignoring file", "path", relPath)
			return nil
		}

		files = append(files, path)
		return nil
	})
	if err != nil {
		return fmt.Errorf("error walking directory: %w", err)
	}

	logger.Info("Found files", "count", len(files))

	// Process files concurrently
	var wg sync.WaitGroup
	concurrencyLimit := 2
	sem := make(chan struct{}, concurrencyLimit)
	var errs []error
	var errMu sync.Mutex

	for _, path := range files {
		sem <- struct{}{} // Acquire semaphore
		wg.Add(1)

		go func(filePath string) {
			defer func() {
				<-sem // Release semaphore
				wg.Done()
			}()

			if err := processFile(filePath, docDir, kvDB, store, defaultHandler, goHandler, llm, logger); err != nil {
				errMu.Lock()
				errs = append(errs, fmt.Errorf("error processing file %s: %w", filePath, err))
				errMu.Unlock()
			}
		}(path)
	}

	wg.Wait()

	if len(errs) > 0 {
		return errs[0] // Return the first error
	}

	return nil
}

func shouldIgnoreWithMatchers(path string, rootDir string, matchers map[string]*ignore.GitIgnore) bool {
	// Check each directory in the path hierarchy for gitignore matchers
	dir := path
	for {
		dir = filepath.Dir(dir)

		// If we've reached or gone beyond the root, stop
		if dir == rootDir || !strings.HasPrefix(dir, rootDir) {
			break
		}

		// Check if this directory has a matcher
		matcher, ok := matchers[dir]
		if !ok {
			continue
		}

		// Get path relative to this directory
		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			continue
		}

		// Check if matcher ignores this path
		if matcher.MatchesPath(relPath) {
			return true
		}
	}

	// Finally check the root directory's gitignore
	if matcher, ok := matchers[rootDir]; ok {
		relPath, err := filepath.Rel(rootDir, path)
		if err == nil && matcher.MatchesPath(relPath) {
			return true
		}
	}

	return false
}

func processFile(
	path string,
	rootDir string,
	kvDB storage.Bolt,
	store golightrag.Storage,
	defaultHandler handler.Default,
	goHandler handler.Go,
	llm golightrag.LLM,
	logger *slog.Logger,
) error {
	// Read file content
	fileData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	fileContent := string(fileData)

	// Generate a file ID based on the relative path
	relPath, err := filepath.Rel(rootDir, path)
	if err != nil {
		return fmt.Errorf("error determining relative path: %w", err)
	}
	fileID := strings.ReplaceAll(relPath, string(filepath.Separator), "_")

	// Check if file has changed by comparing hash
	shouldInsert, err := checkFileHash(kvDB, fileID, fileContent)
	if err != nil {
		return fmt.Errorf("error checking file hash: %w", err)
	}

	if !shouldInsert {
		logger.Debug("File unchanged, skipping", "path", relPath)
		return nil
	}

	logger.Info("Inserting file", "path", relPath)

	// Determine handler based on file extension
	var docHandler golightrag.DocumentHandler
	ext := filepath.Ext(path)
	if ext == ".go" {
		docHandler = goHandler
	} else {
		docHandler = defaultHandler
	}

	// Insert document
	doc := golightrag.Document{
		ID:      fileID,
		Content: fileContent,
	}

	if err := insert(doc, docHandler, store, llm, logger); err != nil {
		return fmt.Errorf("error inserting document: %w", err)
	}

	// Save new hash
	if err := saveFileHash(kvDB, fileID, fileContent); err != nil {
		return fmt.Errorf("error saving file hash: %w", err)
	}

	return nil
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

func insert(
	doc golightrag.Document,
	docHandler golightrag.DocumentHandler,
	storage golightrag.Storage,
	llm golightrag.LLM,
	logger *slog.Logger,
) error {
	now := time.Now()
	defer func() {
		logger.Info("Inserted document", "id", doc.ID, "duration in milliseconds", time.Since(now).Milliseconds())
	}()

	return golightrag.Insert(doc, docHandler, storage, llm, logger)
}

//nolint:funlen // Interactive demo function is long but breaking it down would reduce readability
func query(
	defaultHandler, goHandler golightrag.QueryHandler,
	store golightrag.Storage,
	llm golightrag.LLM,
	logger *slog.Logger,
) {
	// Track conversation for the RAG system
	convo := make([]golightrag.QueryConversation, 0)

	// Maximum turns to keep in conversation history
	const maxTurns = 10

	for {
		// Ask user to select handler first
		log.Println("Select handler (type the number):")
		log.Println("1. Default Handler - General purpose queries")
		log.Println("2. Go Handler - Go programming language specific queries")
		log.Println("(type 'exit' to exit)")

		reader := bufio.NewReader(os.Stdin)
		handlerChoice, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Error reading input: %v\n", err)
			return
		}
		handlerChoice = strings.TrimSpace(handlerChoice)

		if handlerChoice == "exit" {
			log.Println("Exiting...")
			return
		}

		// Select handler based on user choice
		var selectedHandler golightrag.QueryHandler
		switch handlerChoice {
		case "1":
			selectedHandler = defaultHandler
			log.Println("Using Default Handler")
		case "2":
			selectedHandler = goHandler
			log.Println("Using Go Handler")
		default:
			log.Println("Invalid choice. Using Default Handler")
			selectedHandler = defaultHandler
		}

		logger.Info("Selected handler", "type", fmt.Sprintf("%T", selectedHandler))

		// Now get the actual query
		log.Println("Insert query: (type 'exit' to exit)")
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

		// Query the RAG system with the selected handler
		res, err := golightrag.Query(convo, selectedHandler, store, llm, logger)
		if err != nil {
			log.Printf("Error querying: %v\n", err)
			return
		}

		logger.Info("Query result", "result", res)

		logger.Info("Calling LLM", "query duration in milliseconds", time.Since(now).Milliseconds())

		convoStr := make([]string, len(convo))
		for i, conv := range convo {
			convoStr[i] = conv.String()
		}

		promptData := ragPromptData{
			History:     strings.Join(convoStr, "\n"),
			QueryResult: res.String(),
		}

		buf := strings.Builder{}
		tmpl := template.New("rag-prompt")
		tmpl = template.Must(tmpl.Parse(ragPrompt))
		if err := tmpl.Execute(&buf, promptData); err != nil {
			log.Printf("Error executing template: %v\n", err)
			return
		}
		promptText := buf.String()

		logger.Debug("Prompt text", "prompt", promptText)

		llmResponse, err := llm.Chat([]string{promptText})
		if err != nil {
			log.Printf("Error calling LLM: %v\n", err)
			return
		}

		log.Println("\nAssistant:")
		log.Println(llmResponse)
		log.Println()

		convo = append(convo, golightrag.QueryConversation{
			Role:    golightrag.RoleAssistant,
			Message: llmResponse,
		})
	}
}
