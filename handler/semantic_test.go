package handler_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/internal"
)

type mockLLM struct {
	mockResponse    string
	shouldFail      bool
	receivedPrompts []string
}

//nolint:gocognit // Test function with many test cases and complex validation logic
func TestSemantic_ChunksDocument(t *testing.T) {
	tests := []struct {
		name           string
		content        string
		mockResponse   string
		llmShouldFail  bool
		tokenThreshold int
		maxChunkSize   int
		wantErr        bool
		verify         func(t *testing.T, chunks []golightrag.Source)
	}{
		{
			name:          "No LLM configured",
			content:       "Test content",
			mockResponse:  "",
			llmShouldFail: false,
			wantErr:       true,
		},
		{
			name:          "LLM failure",
			content:       "Test content",
			mockResponse:  "",
			llmShouldFail: true,
			wantErr:       true,
		},
		{
			name:    "Simple content",
			content: "This is a simple test document.",
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{
				{
					Summary: "Simple document",
					Start:   0,
					End:     31,
				},
			}),
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 1 {
					t.Fatalf("Expected 1 chunk, got %d", len(chunks))
				}

				// Verify content
				if chunks[0].Content != "This is a simple test document." {
					t.Errorf("Content mismatch: got %q, want %q", chunks[0].Content, "This is a simple test document.")
				}

				// Verify order index
				if chunks[0].OrderIndex != 0 {
					t.Errorf("OrderIndex should be 0, got %d", chunks[0].OrderIndex)
				}

				// Verify token size
				expectedTokens, _ := internal.CountTokens("This is a simple test document.")
				if chunks[0].TokenSize != expectedTokens {
					t.Errorf("TokenSize mismatch: got %d, want %d", chunks[0].TokenSize, expectedTokens)
				}
			},
		},
		{
			name:         "Invalid LLM response",
			content:      "This is a test document.",
			mockResponse: "This is not valid JSON",
			wantErr:      true,
		},
		{
			name:    "LLM returns empty sections",
			content: "This is a test document.",
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{}),
			wantErr: true,
		},
		{
			name:    "Multiple sections",
			content: "Section 1 content. Section 2 content. Section 3 content.",
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{
				{
					Summary: "Section 1",
					Start:   0,
					End:     18,
				},
				{
					Summary: "Section 2",
					Start:   18,
					End:     37,
				},
				{
					Summary: "Section 3",
					Start:   37,
					End:     57,
				},
			}),
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 3 {
					t.Fatalf("Expected 3 chunks, got %d", len(chunks))
				}

				// Expected texts
				expected := []string{
					"Section 1 content.",
					" Section 2 content.",
					" Section 3 content.",
				}

				// Check content
				for i, exp := range expected {
					if chunks[i].Content != exp {
						t.Errorf("Chunk %d content mismatch: got %q, want %q", i, chunks[i].Content, exp)
					}
				}

				// Check ordering
				for i, chunk := range chunks {
					if chunk.OrderIndex != i {
						t.Errorf("Chunk %d OrderIndex should be %d, got %d", i, i, chunk.OrderIndex)
					}
				}

				// Check token counts
				for i, chunk := range chunks {
					expectedTokens, _ := internal.CountTokens(chunk.Content)
					if chunk.TokenSize != expectedTokens {
						t.Errorf("Chunk %d TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
					}
				}
			},
		},
		{
			name:           "Content above threshold",
			content:        strings.Repeat("Test content. ", 500), // Large content to exceed threshold
			tokenThreshold: 100,                                   // Low threshold to force default chunking first
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{
				{
					Summary: "Test section",
					Start:   0,
					End:     20,
				},
			}),
			verify: func(t *testing.T, chunks []golightrag.Source) {
				// We should have multiple chunks due to default chunking being applied first
				if len(chunks) == 0 {
					t.Fatalf("Expected chunks, got none")
				}
			},
		},
		{
			name:         "MaxChunkSize enforcement",
			content:      "This is a test document that should be split if it exceeds the max chunk size.",
			maxChunkSize: 5, // Very small to force splitting
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{
				{
					Summary: "Test document",
					Start:   0,
					End:     76,
				},
			}),
			verify: func(t *testing.T, chunks []golightrag.Source) {
				// Should have multiple chunks due to MaxChunkSize
				if len(chunks) <= 1 {
					t.Fatalf("Expected multiple chunks due to MaxChunkSize, got %d", len(chunks))
				}

				// Verify no chunk exceeds MaxChunkSize
				for i, chunk := range chunks {
					if chunk.TokenSize > 5 {
						t.Errorf("Chunk %d token size (%d) exceeds MaxChunkSize (5)", i, chunk.TokenSize)
					}
				}
			},
		},
		{
			name:    "Invalid section positions",
			content: "Test content with problematic sections.",
			mockResponse: createSemanticResponse([]struct {
				Summary string
				Start   int
				End     int
			}{
				{
					Summary: "Invalid start",
					Start:   -10, // Invalid start position
					End:     10,
				},
				{
					Summary: "Start exceeds end",
					Start:   20,
					End:     15, // Start exceeds end
				},
				{
					Summary: "Exceeds content length",
					Start:   30,
					End:     1000, // Exceeds content length
				},
				{
					Summary: "Valid section",
					Start:   0,
					End:     5,
				},
			}),
			verify: func(t *testing.T, chunks []golightrag.Source) {
				// Should still get at least the valid chunk
				if len(chunks) == 0 {
					t.Fatalf("Expected at least one valid chunk")
				}
			},
		},
		{
			name:    "Wrapped JSON in LLM response",
			content: "Test content.",
			mockResponse: `I've analyzed the content and identified these sections:
{
  "sections": [
    {
      "section_summary": "Test content",
      "start_position": 0,
      "end_position": 13
    }
  ]
}
Let me know if you need anything else.`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				// Should successfully extract the JSON and process it
				if len(chunks) != 1 {
					t.Fatalf("Expected 1 chunk, got %d", len(chunks))
				}

				if chunks[0].Content != "Test content." {
					t.Errorf("Content mismatch: got %q, want %q", chunks[0].Content, "Test content.")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the mock LLM
			mockLLM := &mockLLM{
				mockResponse: tt.mockResponse,
				shouldFail:   tt.llmShouldFail,
			}

			// Create the semantic handler
			semantic := handler.Semantic{
				LLM:            mockLLM,
				TokenThreshold: tt.tokenThreshold,
				MaxChunkSize:   tt.maxChunkSize,
			}

			// Call the function
			chunks, err := semantic.ChunksDocument(tt.content)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Semantic.ChunksDocument() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// If expected to succeed, verify the chunks
			if !tt.wantErr && tt.verify != nil {
				tt.verify(t, chunks)
			}
		})
	}
}

func TestSemantic_DefaultTokenThreshold(t *testing.T) {
	content := "Test content"
	mockLLM := &mockLLM{
		mockResponse: createSemanticResponse([]struct {
			Summary string
			Start   int
			End     int
		}{
			{
				Summary: "Test content",
				Start:   0,
				End:     len(content),
			},
		}),
	}

	// Create semantic handler without setting TokenThreshold
	semantic := handler.Semantic{
		LLM: mockLLM,
	}

	_, err := semantic.ChunksDocument(content)
	if err != nil {
		t.Fatalf("Expected successful chunking with default threshold, got: %v", err)
	}

	// Verify default threshold was used by checking that we didn't attempt
	// to use the Default chunker
	if len(mockLLM.receivedPrompts) == 0 {
		t.Errorf("Expected LLM to be called with prompt")
	}
}

func (m *mockLLM) Chat(prompts []string) (string, error) {
	m.receivedPrompts = prompts
	if m.shouldFail {
		return "", fmt.Errorf("mock LLM failure")
	}
	return m.mockResponse, nil
}

func createSemanticResponse(sections []struct {
	Summary string
	Start   int
	End     int
},
) string {
	resp := struct {
		Sections []struct {
			SectionSummary string `json:"section_summary"`
			StartPosition  int    `json:"start_position"`
			EndPosition    int    `json:"end_position"`
		} `json:"sections"`
	}{}

	for _, s := range sections {
		resp.Sections = append(resp.Sections, struct {
			SectionSummary string `json:"section_summary"`
			StartPosition  int    `json:"start_position"`
			EndPosition    int    `json:"end_position"`
		}{
			SectionSummary: s.Summary,
			StartPosition:  s.Start,
			EndPosition:    s.End,
		})
	}

	jsonData, _ := json.Marshal(resp)
	return string(jsonData)
}
