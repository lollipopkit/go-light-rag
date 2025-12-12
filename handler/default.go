package handler

import (
	"fmt"
	"strings"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/internal"
)

// Default implements both DocumentHandler and QueryHandler interfaces for RAG operations.
// It provides configurable handling for document chunking, entity extraction, and keyword extraction
// with sensible defaults.
type Default struct {
	ChunkMaxTokenSize     int
	ChunkOverlapTokenSize int

	EntityExtractionGoal     string
	EntityTypes              []string
	Language                 string
	EntityExtractionExamples []golightrag.EntityExtractionPromptExample

	KeywordExtractionGoal     string
	KeywordExtractionExamples []golightrag.KeywordExtractionPromptExample

	Config DocumentConfig
}

// DocumentConfig contains configuration parameters for document processing
// during RAG operations, including retry behavior and token length limits.
type DocumentConfig struct {
	MaxRetries              int
	BackoffDuration         time.Duration
	ConcurrencyCount        int
	GleanCount              int
	MaxSummariesTokenLength int
}

const (
	defaultChunkMaxTokenSize       = 1024
	defaultChunkOverlapTokenSize   = 128
	defaultLanguage                = "English"
	defaultMaxSummariesTokenLength = 1200
	defaultBackoffDuration         = 1 * time.Second
	defaultConcurrencyCount        = 1
)

// ChunksDocument splits a document's content into overlapping chunks of text.
// It uses tiktoken to encode and decode tokens, and returns an array of Source objects.
// Each Source contains a portion of the original text with appropriate metadata.
// It returns an error if encoding or decoding fails.
func (d Default) ChunksDocument(content string) ([]golightrag.Source, error) {
	if content == "" {
		return []golightrag.Source{}, nil
	}

	tokenIDs, err := internal.EncodeStringByTiktoken(content)
	if err != nil {
		return nil, fmt.Errorf("failed to encode string: %w", err)
	}

	maxTokenSize := d.ChunkMaxTokenSize
	if maxTokenSize == 0 {
		maxTokenSize = defaultChunkMaxTokenSize
	}
	overlapTokenSize := d.ChunkOverlapTokenSize
	if overlapTokenSize == 0 {
		overlapTokenSize = defaultChunkOverlapTokenSize
	}

	results := []golightrag.Source{}
	for index, start := 0, 0; start < len(tokenIDs); index, start = index+1, start+maxTokenSize-overlapTokenSize {
		end := min(start+maxTokenSize, len(tokenIDs))

		chunkContent, err := internal.DecodeTokensByTiktoken(tokenIDs[start:end])
		if err != nil {
			return nil, fmt.Errorf("failed to decode tokens: %w", err)
		}

		trimmedContent := strings.TrimSpace(chunkContent)

		tokenCount, err := internal.CountTokens(trimmedContent)
		if err != nil {
			return nil, fmt.Errorf("failed to count tokens: %w", err)
		}

		results = append(results, golightrag.Source{
			Content:    trimmedContent,
			TokenSize:  tokenCount,
			OrderIndex: index,
		})
	}

	return results, nil
}

// EntityExtractionPromptData returns the data needed to generate prompts for extracting
// entities and relationships from text content.
func (d Default) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	goal := d.EntityExtractionGoal
	if goal == "" {
		goal = defaultEntityExtractionGoal
	}
	entityTypes := d.EntityTypes
	if entityTypes == nil {
		entityTypes = defaultEntityTypes
	}
	language := d.Language
	if language == "" {
		language = defaultLanguage
	}
	examples := d.EntityExtractionExamples
	if examples == nil {
		examples = defaultEntityExtractionExamples
	}
	return golightrag.EntityExtractionPromptData{
		Goal:        goal,
		EntityTypes: entityTypes,
		Language:    language,
		Examples:    examples,
	}
}

// MaxRetries returns the maximum number of retry attempts for RAG operations
// as configured in the DocumentConfig.
func (d Default) MaxRetries() int {
	return d.Config.MaxRetries
}

// BackoffDuration returns the backoff duration between retries for RAG operations
// as configured in the DocumentConfig.
func (d Default) BackoffDuration() time.Duration {
	if d.Config.BackoffDuration == 0 {
		return defaultBackoffDuration
	}
	return d.Config.BackoffDuration
}

// ConcurrencyCount returns the number of concurrent requests to the LLM
// as configured in the DocumentConfig.
func (d Default) ConcurrencyCount() int {
	if d.Config.ConcurrencyCount == 0 {
		return defaultConcurrencyCount
	}
	return d.Config.ConcurrencyCount
}

// GleanCount returns the number of sources to extract during RAG operations
// as configured in the DocumentConfig.
func (d Default) GleanCount() int {
	return d.Config.GleanCount
}

// MaxSummariesTokenLength returns the maximum token length for summaries.
// If not explicitly configured, it returns the default value.
func (d Default) MaxSummariesTokenLength() int {
	if d.Config.MaxSummariesTokenLength == 0 {
		return defaultMaxSummariesTokenLength
	}
	return d.Config.MaxSummariesTokenLength
}

// KeywordExtractionPromptData returns the data needed to generate prompts for extracting
// keywords from user queries and conversation history.
func (d Default) KeywordExtractionPromptData() golightrag.KeywordExtractionPromptData {
	goal := d.KeywordExtractionGoal
	if goal == "" {
		goal = defaultKeywordExtractionGoal
	}
	examples := d.KeywordExtractionExamples
	if examples == nil {
		examples = defaultKeywordExtractionExamples
	}
	return golightrag.KeywordExtractionPromptData{
		Goal:     goal,
		Examples: examples,
	}
}
