package handler_test

import (
	"strings"
	"testing"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/internal"
)

//nolint:gocognit // Test function with many test cases and conditional logic
func TestDefault_ChunksDocument(t *testing.T) {
	tests := []struct {
		name             string
		content          string
		handlerConfig    handler.Default
		expectedChunks   int
		expectError      bool
		verificationFunc func(t *testing.T, chunks []golightrag.Source)
	}{
		{
			name:           "Empty content",
			content:        "",
			expectedChunks: 0,
		},
		{
			name:           "Small content within single chunk",
			content:        "This is a small text that should fit in a single chunk.",
			expectedChunks: 1,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				verifyChunkCount(t, chunks, 1)
				if chunks[0].OrderIndex != 0 {
					t.Errorf("Expected OrderIndex 0, got %d", chunks[0].OrderIndex)
				}
				if chunks[0].Content != "This is a small text that should fit in a single chunk." {
					t.Errorf("Content mismatch: %s", chunks[0].Content)
				}
			},
		},
		{
			name:    "Content that spans multiple chunks",
			content: strings.Repeat("This sentence contains about nine tokens. ", 300), // ~2600 tokens
			// With default 1024 max size, should create ~3 chunks
			expectedChunks: 3,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) < 2 {
					t.Fatalf("Expected multiple chunks, got %d", len(chunks))
				}

				verifyOrderIndexes(t, chunks)
				verifyChunkOverlap(t, chunks)
			},
		},
		{
			name:    "Custom chunk size",
			content: strings.Repeat("Short text. ", 30), // ~60-90 tokens
			handlerConfig: handler.Default{
				ChunkMaxTokenSize:     30, // Small chunk size
				ChunkOverlapTokenSize: 5,  // Small overlap
			},
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) < 3 {
					t.Fatalf("Expected at least 3 chunks with small chunk size, got %d", len(chunks))
				}

				// Check token sizes don't exceed max
				for i, chunk := range chunks {
					if chunk.TokenSize > 30 {
						t.Errorf("Chunk %d has TokenSize %d, expected <= 30", i, chunk.TokenSize)
					}
				}
			},
		},
		{
			name:    "Unicode and special characters",
			content: "Special characters: 🚀 😊 üñîçødé\nNew lines\tTabs中文日本語",
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				verifyChunkCount(t, chunks, 1)

				// Check all special characters are preserved
				if !strings.Contains(chunks[0].Content, "🚀") ||
					!strings.Contains(chunks[0].Content, "üñîçødé") ||
					!strings.Contains(chunks[0].Content, "中文") {
					t.Errorf("Special characters not preserved: %s", chunks[0].Content)
				}
			},
		},
	}

	runChunksDocumentTests(t, tests)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use provided handler config or default
			h := tt.handlerConfig

			chunks, err := h.ChunksDocument(tt.content)

			// Check error expectations
			if (err != nil) != tt.expectError {
				t.Errorf("ChunksDocument() error = %v, expectError = %v", err, tt.expectError)
				return
			}

			if tt.expectError {
				return
			}

			// Check number of chunks if expected
			if tt.expectedChunks > 0 && len(chunks) != tt.expectedChunks {
				t.Errorf("Expected %d chunks, got %d", tt.expectedChunks, len(chunks))
			}

			// Verify token sizes make sense
			for i, chunk := range chunks {
				// Token size should be positive for non-empty chunks
				if len(chunk.Content) > 0 && chunk.TokenSize <= 0 {
					t.Errorf("Chunk %d has invalid token size: %d", i, chunk.TokenSize)
				}

				// Verify token counts match content
				expectedTokens, err := internal.CountTokens(chunk.Content)
				if err != nil {
					t.Errorf("Failed to count tokens for verification: %v", err)
				}
				if chunk.TokenSize != expectedTokens {
					t.Errorf("Chunk %d: reported TokenSize %d doesn't match actual count %d",
						i, chunk.TokenSize, expectedTokens)
				}
			}

			// Run custom verification if provided
			if tt.verificationFunc != nil {
				tt.verificationFunc(t, chunks)
			}
		})
	}
}

func verifyChunkCount(t *testing.T, chunks []golightrag.Source, expected int) {
	t.Helper()
	if len(chunks) != expected {
		t.Fatalf("Expected %d chunk(s), got %d", expected, len(chunks))
	}
}

func verifyOrderIndexes(t *testing.T, chunks []golightrag.Source) {
	t.Helper()
	for i, chunk := range chunks {
		if chunk.OrderIndex != i {
			t.Errorf("Chunk %d has OrderIndex %d, expected %d", i, chunk.OrderIndex, i)
		}
	}
}

func verifyTokenCounts(t *testing.T, chunks []golightrag.Source) {
	t.Helper()
	for i, chunk := range chunks {
		if len(chunk.Content) > 0 && chunk.TokenSize <= 0 {
			t.Errorf("Chunk %d has invalid token size: %d", i, chunk.TokenSize)
		}

		expectedTokens, err := internal.CountTokens(chunk.Content)
		if err != nil {
			t.Errorf("Failed to count tokens for verification: %v", err)
		} else if chunk.TokenSize != expectedTokens {
			t.Errorf("Chunk %d: reported TokenSize %d doesn't match actual count %d",
				i, chunk.TokenSize, expectedTokens)
		}
	}
}

func verifyChunkOverlap(t *testing.T, chunks []golightrag.Source) {
	t.Helper()
	if len(chunks) >= 2 {
		firstChunkEnd := chunks[0].Content[len(chunks[0].Content)-20:]
		secondChunkStart := chunks[1].Content[:20]

		if !strings.Contains(chunks[1].Content, firstChunkEnd) {
			t.Errorf("Expected overlap between chunks not found")
		}

		t.Logf("Chunk transition: '...%s' -> '%s...'", firstChunkEnd, secondChunkStart)
	}
}

func runChunksDocumentTests(t *testing.T, tests []struct {
	name             string
	content          string
	handlerConfig    handler.Default
	expectedChunks   int
	expectError      bool
	verificationFunc func(t *testing.T, chunks []golightrag.Source)
},
) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use provided handler config or default
			h := tt.handlerConfig

			chunks, err := h.ChunksDocument(tt.content)

			// Check error expectations
			if (err != nil) != tt.expectError {
				t.Errorf("ChunksDocument() error = %v, expectError = %v", err, tt.expectError)
				return
			}

			if tt.expectError {
				return
			}

			// Check number of chunks if expected
			if tt.expectedChunks > 0 && len(chunks) != tt.expectedChunks {
				t.Errorf("Expected %d chunks, got %d", tt.expectedChunks, len(chunks))
			}

			// Verify token sizes make sense
			verifyTokenCounts(t, chunks)

			// Run custom verification if provided
			if tt.verificationFunc != nil {
				tt.verificationFunc(t, chunks)
			}
		})
	}
}
