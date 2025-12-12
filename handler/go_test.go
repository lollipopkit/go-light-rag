package handler_test

import (
	"strings"
	"testing"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/handler"
	"github.com/lollipopkit/go-light-rag/internal"
)

//nolint:gocognit,cyclop,gocyclo // Test cases are complex
func TestGo_ChunksDocument(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
		verify  func(t *testing.T, chunks []golightrag.Source)
	}{
		{
			name:    "Empty content",
			content: "",
			wantErr: true,
		},
		{
			name:    "Invalid Go code",
			content: "this is not valid Go code",
			wantErr: true,
		},
		{
			name:    "Package declaration only",
			content: `package example`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 1 {
					t.Fatalf("Expected 1 chunk, got %d", len(chunks))
				}

				expectedContent := "package example"
				if chunks[0].Content != expectedContent {
					t.Errorf("Content mismatch: got %q, want %q", chunks[0].Content, expectedContent)
				}

				expectedTokens, _ := internal.CountTokens(expectedContent)
				if chunks[0].TokenSize != expectedTokens {
					t.Errorf("TokenSize mismatch: got %d, want %d", chunks[0].TokenSize, expectedTokens)
				}

				if chunks[0].OrderIndex != 0 {
					t.Errorf("OrderIndex should be 0, got %d", chunks[0].OrderIndex)
				}
			},
		},
		{
			name: "Package and imports",
			content: `package example

import (
	"fmt"
	"strings"
)`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 1 {
					t.Fatalf("Expected 1 chunk, got %d", len(chunks))
				}

				expectedTokens, _ := internal.CountTokens(chunks[0].Content)
				if chunks[0].TokenSize != expectedTokens {
					t.Errorf("TokenSize mismatch: got %d, want %d", chunks[0].TokenSize, expectedTokens)
				}

				if !strings.Contains(chunks[0].Content, "import") {
					t.Errorf("Chunk should contain imports")
				}
			},
		},
		{
			name: "Simple function",
			content: `package example

func Add(a, b int) int {
	return a + b
}`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 2 {
					t.Fatalf("Expected 2 chunks, got %d", len(chunks))
				}

				// Verify package declaration
				if !strings.Contains(chunks[0].Content, "package example") {
					t.Errorf("First chunk should be package declaration")
				}

				// Verify function chunk
				if !strings.Contains(chunks[1].Content, "func Add") {
					t.Errorf("Second chunk should contain the function")
				}

				// Verify token counts
				for i, chunk := range chunks {
					expectedTokens, _ := internal.CountTokens(chunk.Content)
					if chunk.TokenSize != expectedTokens {
						t.Errorf("Chunk %d: TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
					}
				}

				// Verify order index
				for i, chunk := range chunks {
					if chunk.OrderIndex != i {
						t.Errorf("Chunk %d: OrderIndex should be %d, got %d", i, i, chunk.OrderIndex)
					}
				}
			},
		},
		{
			name: "Type definition",
			content: `package example

type Person struct {
	Name string
	Age  int
}`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 2 {
					t.Fatalf("Expected 2 chunks, got %d", len(chunks))
				}

				// Verify package declaration
				if !strings.Contains(chunks[0].Content, "package example") {
					t.Errorf("First chunk should be package declaration")
				}

				// Verify type definition
				if !strings.Contains(chunks[1].Content, "type Person struct") {
					t.Errorf("Second chunk should contain the type definition")
				}

				// Verify each chunk has package declaration
				for i, chunk := range chunks {
					if !strings.Contains(chunk.Content, "package example") {
						t.Errorf("Chunk %d missing package declaration", i)
					}

					expectedTokens, _ := internal.CountTokens(chunk.Content)
					if chunk.TokenSize != expectedTokens {
						t.Errorf("Chunk %d: TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
					}
				}
			},
		},
		{
			name: "Constants and variables",
			content: `package example

const (
	MaxAge = 120
	MinAge = 0
)

var DefaultName = "Anonymous"`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) != 3 {
					t.Fatalf("Expected 3 chunks, got %d", len(chunks))
				}

				// Verify const declaration exists
				foundConst := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "const (") {
						foundConst = true
						break
					}
				}
				if !foundConst {
					t.Errorf("No chunk contains const declaration")
				}

				// Verify var declaration exists
				foundVar := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "var DefaultName") {
						foundVar = true
						break
					}
				}
				if !foundVar {
					t.Errorf("No chunk contains var declaration")
				}

				// Verify token counts and package declaration
				for i, chunk := range chunks {
					if !strings.Contains(chunk.Content, "package example") {
						t.Errorf("Chunk %d missing package declaration", i)
					}

					expectedTokens, _ := internal.CountTokens(chunk.Content)
					if chunk.TokenSize != expectedTokens {
						t.Errorf("Chunk %d: TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
					}
				}
			},
		},
		{
			name: "Complete example",
			content: `package example

import (
	"fmt"
	"strings"
)

const (
	DefaultPrefix = "User-"
)

var MaxUserCount = 100

type User struct {
	ID   int
	Name string
}

func (u *User) FullName() string {
	return DefaultPrefix + u.Name
}

func CreateUser(name string) User {
	return User{
		Name: name,
	}
}`,
			verify: func(t *testing.T, chunks []golightrag.Source) {
				// Should have 6 chunks: header+imports, const, var, type, method, function
				if len(chunks) != 6 {
					t.Fatalf("Expected 6 chunks, got %d", len(chunks))
				}

				// Verify order indexes are sequential
				for i, chunk := range chunks {
					if chunk.OrderIndex != i {
						t.Errorf("Expected OrderIndex %d, got %d", i, chunk.OrderIndex)
					}
				}

				// Check that each chunk contains the package name
				for i, chunk := range chunks {
					if !strings.Contains(chunk.Content, "package example") {
						t.Errorf("Chunk %d doesn't contain package declaration", i)
					}

					expectedTokens, _ := internal.CountTokens(chunk.Content)
					if chunk.TokenSize != expectedTokens {
						t.Errorf("Chunk %d: TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
					}
				}

				// Verify specific chunks exist
				chunkTypes := map[string]bool{
					"import":           false,
					"const":            false,
					"var MaxUserCount": false,
					"type User struct": false,
					"func (u *User)":   false,
					"func CreateUser":  false,
				}

				for _, chunk := range chunks {
					for key := range chunkTypes {
						if strings.Contains(chunk.Content, key) {
							chunkTypes[key] = true
						}
					}
				}

				for key, found := range chunkTypes {
					if !found {
						t.Errorf("No chunk contains %q", key)
					}
				}
			},
		},
	}

	runGoChunksDocumentTests(t, tests)
}

func TestGo_ChunksDocument_WithComments(t *testing.T) {
	goHandler := handler.Go{}

	// Test with comments
	codeWithComments := `package example

// Add adds two integers and returns the result
// It demonstrates basic addition
func Add(a, b int) int {
	return a + b
}`

	chunks, err := goHandler.ChunksDocument(codeWithComments)
	if err != nil {
		t.Fatalf("Failed to chunk document: %v", err)
	}

	if len(chunks) != 2 {
		t.Errorf("Expected 2 chunks, got %d", len(chunks))
	}

	// Verify the function chunk contains the comments
	if len(chunks) > 1 && !strings.Contains(chunks[1].Content, "Add adds two integers") {
		t.Errorf("Function chunk should contain comments")
	}

	// Verify token counts
	for i, chunk := range chunks {
		expectedTokens, err := internal.CountTokens(chunk.Content)
		if err != nil {
			t.Errorf("Failed to count tokens: %v", err)
		}
		if chunk.TokenSize != expectedTokens {
			t.Errorf("Chunk %d: TokenSize mismatch: got %d, want %d", i, chunk.TokenSize, expectedTokens)
		}
	}
}

func runGoChunksDocumentTests(t *testing.T, tests []struct {
	name    string
	content string
	wantErr bool
	verify  func(t *testing.T, chunks []golightrag.Source)
},
) {
	goHandler := handler.Go{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := goHandler.ChunksDocument(tt.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("Go.ChunksDocument() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			if tt.verify != nil {
				tt.verify(t, got)
			}
		})
	}
}
