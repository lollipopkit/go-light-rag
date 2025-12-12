package handler

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/lollipopkit/go-light-rag/internal"
)

// Go implements specialized document handling for Go source code.
// It extends the Default handler with Go-specific functionality for parsing
// and processing Go source files during RAG operations.
type Go struct {
	Default
}

func getCodeBetweenLines(content string, start, end int) string {
	lines := strings.Split(content, "\n")
	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}

	return strings.Join(lines[start-1:end], "\n")
}

// ChunksDocument splits Go source code into semantically meaningful chunks.
// It parses the Go code using Go's AST parser and divides it into logical sections:
// - Package declaration and imports as one chunk
// - Each function or method as individual chunks
// - Type declarations (structs, interfaces) as individual chunks
// - Constants and variables as separate chunks
//
// Each chunk includes its package declaration to ensure it can be parsed independently.
// It returns an array of Source objects, each containing a portion of the original code
// with appropriate metadata including token size and order index.
// It returns an error if parsing fails or token counting encounters issues.
//
//nolint:gocognit,funlen // Go AST parsing function with necessary conditional logic for different node types
func (g Go) ChunksDocument(content string) ([]golightrag.Source, error) {
	// Parse the Go file
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", content, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Go code: %w", err)
	}

	var chunks []golightrag.Source
	orderIndex := 0

	// Add chunk for package declaration and imports
	importEnd := 0
	if len(file.Imports) > 0 {
		lastImport := file.Imports[len(file.Imports)-1]
		importEnd = fset.Position(lastImport.End()).Line
	}

	// Create the package and imports chunk
	headerCode := getCodeBetweenLines(content, 1, importEnd+1)
	headerTokenSize, err := internal.CountTokens(headerCode)
	if err != nil {
		return nil, fmt.Errorf("failed to count tokens on header: %w", err)
	}
	if len(headerCode) > 0 {
		chunks = append(chunks, golightrag.Source{
			Content:    headerCode,
			TokenSize:  headerTokenSize,
			OrderIndex: orderIndex,
		})
		orderIndex++
	}

	// Get package prefix to add to other chunks
	packagePrefix := fmt.Sprintf("package %s\n\n", file.Name.Name)

	// Process declarations (types, functions, methods, etc.)
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			// Handle function/method declaration
			startPos := fset.Position(d.Pos())
			endPos := fset.Position(d.End())

			// Get associated comments
			var comments string
			if d.Doc != nil {
				comments = d.Doc.Text()
			}

			functionCode := getCodeBetweenLines(content, startPos.Line, endPos.Line)
			// Add comments and package prefix
			functionCode = packagePrefix + comments + functionCode

			tokenSize, err := internal.CountTokens(functionCode)
			if err != nil {
				return nil, fmt.Errorf("failed to count tokens on function: %w", err)
			}
			chunks = append(chunks, golightrag.Source{
				Content:    functionCode,
				TokenSize:  tokenSize,
				OrderIndex: orderIndex,
			})
			orderIndex++

		case *ast.GenDecl:
			//nolint:exhaustive // Ignore other declaration types
			switch d.Tok {
			case token.TYPE:
				// Handle type declarations (structs, interfaces)
				for range d.Specs {
					startPos := fset.Position(d.Pos())
					endPos := fset.Position(d.End())

					// Get associated comments
					var comments string
					if d.Doc != nil {
						comments = d.Doc.Text()
					}

					typeCode := getCodeBetweenLines(content, startPos.Line, endPos.Line)
					// Add package prefix
					typeCode = packagePrefix + comments + typeCode

					tokenSize, err := internal.CountTokens(typeCode)
					if err != nil {
						return nil, fmt.Errorf("failed to count tokens on type: %w", err)
					}
					chunks = append(chunks, golightrag.Source{
						Content:    typeCode,
						TokenSize:  tokenSize,
						OrderIndex: orderIndex,
					})
					orderIndex++
				}
			case token.CONST, token.VAR:
				// Group constants and variables
				startPos := fset.Position(d.Pos())
				endPos := fset.Position(d.End())

				declCode := getCodeBetweenLines(content, startPos.Line, endPos.Line)
				// Add package prefix
				declCode = packagePrefix + declCode

				tokenSize, err := internal.CountTokens(declCode)
				if err != nil {
					return nil, fmt.Errorf("failed to count tokens on declaration: %w", err)
				}
				chunks = append(chunks, golightrag.Source{
					Content:    declCode,
					TokenSize:  tokenSize,
					OrderIndex: orderIndex,
				})
				orderIndex++
			default:
				// Ignore other declaration types
				continue
			}
		}
	}

	return chunks, nil
}

// EntityExtractionPromptData returns the data needed to generate prompts for extracting
// entities and relationships from Go source code content.
// It provides Go-specific entity extraction configurations, including custom goals,
// entity types, and examples tailored for Go language parsing.
func (g Go) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	language := g.Language
	if language == "" {
		language = defaultLanguage
	}
	return golightrag.EntityExtractionPromptData{
		Goal:        goEntityExtractionGoal,
		EntityTypes: goEntityTypes,
		Language:    language,
		Examples:    goEntityExtractionExamples,
	}
}

// KeywordExtractionPromptData returns the data needed to generate prompts for extracting
// keywords from Go source code and related queries.
// It provides Go-specific keyword extraction configurations with custom goals
// and examples optimized for Go language patterns.
func (g Go) KeywordExtractionPromptData() golightrag.KeywordExtractionPromptData {
	return golightrag.KeywordExtractionPromptData{
		Goal:     goKeywordExtractionGoal,
		Examples: goKeywordExtractionExamples,
	}
}
