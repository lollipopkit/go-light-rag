package handler

import (
	"encoding/json"
	"fmt"
	"strings"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/internal"
)

// Semantic implements document handling with semantically meaningful chunking.
// It extends the Default handler and leverages an LLM to create chunks based on
// natural content divisions rather than fixed token counts.
// This results in more coherent chunks that preserve semantic relationships
// within the text, improving RAG quality at the cost of additional LLM calls.
type Semantic struct {
	Default

	// LLM is the language model to use for semantic chunking.
	// This field is required and must be set before using the handler.
	LLM golightrag.LLM

	// TokenThreshold is the maximum number of tokens that can be sent to the LLM
	// in a single request. Documents larger than this threshold will be pre-chunked
	// using the Default chunker before semantic processing. Defaults to 8000 if not set.
	TokenThreshold int

	// MaxChunkSize defines the maximum token size for any individual semantic chunk.
	// If a semantic section exceeds this size, it will be further divided using
	// the Default chunker. If set to 0, no maximum size is enforced.
	MaxChunkSize int
}

type sectionInfo struct {
	SectionSummary string `json:"section_summary"`
	StartPosition  int    `json:"start_position"`
	EndPosition    int    `json:"end_position"`
}

type semanticChunkResponse struct {
	Sections []sectionInfo `json:"sections"`
}

const defaultSemanticTokenthreshold = 8000

// ChunksDocument splits a document's content into semantically meaningful chunks
// using the configured LLM to identify natural content boundaries.
//
// For documents smaller than TokenThreshold, it processes the entire content directly.
// For larger documents, it first applies Default chunking and then semantically
// processes each chunk separately.
//
// The method preserves document ordering by assigning appropriate OrderIndex values
// to each chunk. It falls back to Default chunking when semantic chunking fails
// or produces no valid chunks.
//
// It returns an array of Source objects, each containing a semantically coherent
// portion of the original text with appropriate metadata.
// It returns an error if the LLM is not configured, the LLM call fails,
// or token counting encounters issues.
func (s Semantic) ChunksDocument(content string) ([]golightrag.Source, error) {
	if s.LLM == nil {
		return nil, fmt.Errorf("LLM is required for semantic chunking")
	}

	if s.TokenThreshold == 0 {
		s.TokenThreshold = defaultSemanticTokenthreshold
	}

	tokenCount, err := internal.CountTokens(content)
	if err != nil {
		return nil, fmt.Errorf("failed to count tokens: %w", err)
	}

	// If content is too large, fall back to the Default chunking method
	if tokenCount > s.TokenThreshold {
		// Split the content into manageable pieces first
		defaultChunks, err := s.Default.ChunksDocument(content)
		if err != nil {
			return nil, fmt.Errorf("failed to pre-chunk large content: %w", err)
		}

		// Process each large chunk semantically and combine the results
		var allSources []golightrag.Source
		for i, chunk := range defaultChunks {
			sources, err := s.semanticChunk(chunk.Content)
			if err != nil {
				// If semantic chunking fails, use the original chunk
				allSources = append(allSources, golightrag.Source{
					Content:    chunk.Content,
					TokenSize:  chunk.TokenSize,
					OrderIndex: len(allSources),
				})
				continue
			}

			// Adjust the order indices and add to the combined result
			for j := range sources {
				sources[j].OrderIndex = i*100 + j
			}
			allSources = append(allSources, sources...)
		}
		return allSources, nil
	}

	// For reasonably sized content, process it directly
	return s.semanticChunk(content)
}

//nolint:gocognit // Semantic chunking function with LLM parsing and validation logic
func (s Semantic) semanticChunk(content string) ([]golightrag.Source, error) {
	// Prepare the prompt with the content
	prompt := strings.ReplaceAll(semanticChunkingPrompt, "{{.Content}}", content)

	// Call the LLM to generate the semantic chunks
	response, err := s.LLM.Chat([]string{prompt})
	if err != nil {
		return nil, fmt.Errorf("failed to generate semantic chunks: %w", err)
	}

	// Parse the LLM response
	var semanticResponse semanticChunkResponse
	if err := json.Unmarshal([]byte(response), &semanticResponse); err != nil {
		// If JSON parsing fails, try to extract JSON from the response
		jsonStart := strings.Index(response, "{")
		jsonEnd := strings.LastIndex(response, "}")
		if jsonStart >= 0 && jsonEnd > jsonStart {
			jsonStr := response[jsonStart : jsonEnd+1]
			if err := json.Unmarshal([]byte(jsonStr), &semanticResponse); err != nil {
				return nil, fmt.Errorf("failed to parse semantic chunks response: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to parse semantic chunks response: %w", err)
		}
	}

	// Validate the sections
	if len(semanticResponse.Sections) == 0 {
		return nil, fmt.Errorf("LLM did not identify any semantic sections")
	}

	// Convert the sections to Source objects
	sources := make([]golightrag.Source, 0, len(semanticResponse.Sections))
	for i, section := range semanticResponse.Sections {
		// Ensure start and end positions are valid
		if section.StartPosition < 0 {
			section.StartPosition = 0
		}
		if section.EndPosition > len(content) {
			section.EndPosition = len(content)
		}
		if section.StartPosition >= section.EndPosition {
			// Skip invalid sections
			continue
		}

		// Extract the section text
		sectionText := content[section.StartPosition:section.EndPosition]

		// Skip empty sections
		if len(strings.TrimSpace(sectionText)) == 0 {
			continue
		}

		// Count tokens for the section
		tokenCount, err := internal.CountTokens(sectionText)
		if err != nil {
			return nil, fmt.Errorf("failed to count tokens for section: %w", err)
		}

		// Apply max chunk size if specified
		if s.MaxChunkSize > 0 && tokenCount > s.MaxChunkSize {
			// Create a temporary Default handler with appropriate settings based on MaxChunkSize
			tempDefault := Default{
				ChunkMaxTokenSize:     s.MaxChunkSize,
				ChunkOverlapTokenSize: min(s.MaxChunkSize/4, 20), // Reasonable overlap that won't exceed MaxChunkSize
			}

			// If a section is too large, further split it using the Default chunker
			defaultSources, err := tempDefault.ChunksDocument(sectionText)
			if err != nil {
				return nil, fmt.Errorf("failed to apply default chunking to large section: %w", err)
			}

			// Adjust the order index for the sub-chunks
			for j := range defaultSources {
				defaultSources[j].OrderIndex = i*100 + j
			}

			sources = append(sources, defaultSources...)
		} else {
			// Add the section as a single chunk
			sources = append(sources, golightrag.Source{
				Content:    sectionText,
				TokenSize:  tokenCount,
				OrderIndex: i,
			})
		}
	}

	// If no valid sources were created, fall back to the Default chunker
	if len(sources) == 0 {
		defaultSources, err := s.Default.ChunksDocument(content)
		if err != nil {
			return nil, fmt.Errorf("failed to apply default chunking after semantic chunking failed: %w", err)
		}
		return defaultSources, nil
	}

	return sources, nil
}
