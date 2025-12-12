package golightrag_test

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"

	golightrag "github.com/lollipopkit/go-light-rag"
)

//nolint:gocognit,gocyclo,cyclop // Ignore complexity for this test
func TestInsert(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	// logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
	// 	Level: slog.LevelDebug,
	// }))

	t.Run("Successful insertion", func(t *testing.T) {
		doc := golightrag.Document{
			ID:      "test-doc-1",
			Content: "Test content with null byte \x00 and spaces   ",
		}

		// Create mock LLM that returns valid entity and relationship extraction
		mockLLM := &MockLLM{
			chatResponse: `
{
  "entities": [
    {
      "entity_name": "ENTITY1",
      "entity_type": "PERSON",
      "entity_description": "This is a description of Entity1"
    },
    {
      "entity_name": "ENTITY2",
      "entity_type": "ORGANIZATION",
      "entity_description": "This is a description of Entity2"
    }
  ],
  "relationships": [
    {
      "source_entity": "ENTITY1", 
      "target_entity": "ENTITY2",
      "relationship_description": "Entity1 is related to Entity2",
      "relationship_keywords": ["RELATED_TO", "RELATED", "TO"],
      "relationship_strength": 1.0
    }
  ]
}`,
			chatCalls: make([][]string, 0),
		}

		// Create mock handler
		handler := &MockDocumentHandler{
			sources: []golightrag.Source{
				{
					Content:    "Test content",
					TokenSize:  2,
					OrderIndex: 0,
				},
			},
			entityExtractionPromptData: golightrag.EntityExtractionPromptData{
				Goal:        "Extract entities",
				EntityTypes: []string{"PERSON", "ORGANIZATION"},
				Language:    "English",
			},
			maxRetries:  3,
			gleanCount:  2,
			maxTokenLen: 1000,
		}

		// Create mock storage
		storage := &MockStorage{
			entities:      make(map[string]golightrag.GraphEntity),
			relationships: make(map[string]golightrag.GraphRelationship),
		}

		// Call the function under test
		err := golightrag.Insert(doc, handler, storage, mockLLM, logger)
		// Assertions
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		if !storage.kvUpsertSourcesCalled {
			t.Error("Expected KVUpsertSources to be called")
		}

		if !storage.graphUpsertEntityCalled {
			t.Error("Expected GraphUpsertEntity to be called")
		}

		if !storage.vectorUpsertEntityCalled {
			t.Error("Expected VectorUpsertEntity to be called")
		}

		if !storage.graphUpsertRelationshipCalled {
			t.Error("Expected GraphUpsertRelationship to be called")
		}

		if !storage.vectorUpsertRelationshipCalled {
			t.Error("Expected VectorUpsertRelationship to be called")
		}

		// Verify that entities were extracted and stored
		if len(storage.entities) != 2 {
			t.Errorf("Expected 2 entities, got %d %+v", len(storage.entities), storage.entities)
		}

		// Check entity1 values
		entity1, exists := storage.entities["ENTITY1"]
		if !exists {
			t.Error("Expected ENTITY1 to be stored")
		} else {
			if entity1.Type != "PERSON" {
				t.Errorf("Expected ENTITY1 type to be PERSON, got %s", entity1.Type)
			}
			if !strings.Contains(entity1.Descriptions, "description of Entity1") {
				t.Errorf("Expected ENTITY1 description to contain 'description of Entity1', got %s", entity1.Descriptions)
			}
			// Check source ID
			expectedSourceID := fmt.Sprintf("%s-chunk-0", doc.ID)
			if !strings.Contains(entity1.SourceIDs, expectedSourceID) {
				t.Errorf("Expected source ID %s in entity SourceIDs: %s", expectedSourceID, entity1.SourceIDs)
			}
		}

		// Check entity2 values
		entity2, exists := storage.entities["ENTITY2"]
		if !exists {
			t.Error("Expected ENTITY2 to be stored")
		} else {
			if entity2.Type != "ORGANIZATION" {
				t.Errorf("Expected ENTITY2 type to be ORGANIZATION, got %s", entity2.Type)
			}
			if !strings.Contains(entity2.Descriptions, "description of Entity2") {
				t.Errorf("Expected ENTITY2 description to contain 'description of Entity2', got %s", entity2.Descriptions)
			}
		}

		// Verify that relationships were extracted and stored
		if len(storage.relationships) != 1 {
			t.Errorf("Expected 1 relationship, got %d", len(storage.relationships))
		}

		// Check relationship values
		relKey := "ENTITY1:ENTITY2"
		rel, exists := storage.relationships[relKey]
		if !exists {
			t.Fatalf("Expected ENTITY1-ENTITY2 relationship to be stored")
		}
		if rel.SourceEntity != "ENTITY1" {
			t.Errorf("Expected relationship source to be ENTITY1, got %s", rel.SourceEntity)
		}
		if rel.TargetEntity != "ENTITY2" {
			t.Errorf("Expected relationship target to be ENTITY2, got %s", rel.TargetEntity)
		}
		if !strings.Contains(rel.Descriptions, "Entity1 is related to Entity2") {
			t.Errorf("Expected relationship description to contain 'Entity1 is related to Entity2', got %s", rel.Descriptions)
		}
		if len(rel.Keywords) != 3 {
			t.Fatalf("Expected relationship keywords to contain 3 keywords, got %d", len(rel.Keywords))
		}
		if !strings.Contains(rel.Keywords[0], "RELATED_TO") {
			t.Errorf("Expected relationship keywords to contain 'RELATED_TO', got %s", rel.Keywords[0])
		}
		if !strings.Contains(rel.Keywords[1], "RELATED") {
			t.Errorf("Expected relationship keywords to contain 'RELATED', got %s", rel.Keywords[1])
		}
		if !strings.Contains(rel.Keywords[2], "TO") {
			t.Errorf("Expected relationship keywords to contain 'TO', got %s", rel.Keywords[2])
		}
		expectedSourceID := fmt.Sprintf("%s-chunk-0", doc.ID)
		if !strings.Contains(rel.SourceIDs, expectedSourceID) {
			t.Errorf("Expected source ID %s in relationship SourceIDs: %s", expectedSourceID, rel.SourceIDs)
		}
	})

	t.Run("Invalid entity extraction format", func(t *testing.T) {
		doc := golightrag.Document{
			ID:      "test-doc-6",
			Content: "Test content",
		}

		// Create mock LLM that returns invalid format
		mockLLM := &MockLLM{
			chatResponse: `This is not a valid format`,
		}

		// Create mock handler
		handler := &MockDocumentHandler{
			sources: []golightrag.Source{
				{
					Content:    "Test content",
					TokenSize:  2,
					OrderIndex: 0,
				},
			},
			entityExtractionPromptData: golightrag.EntityExtractionPromptData{
				Goal:        "Extract entities",
				EntityTypes: []string{"PERSON", "ORGANIZATION"},
				Language:    "English",
			},
			maxRetries:  1,
			gleanCount:  1,
			maxTokenLen: 1000,
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		err := golightrag.Insert(doc, handler, storage, mockLLM, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to invalid format, got nil")
		}
	})

	t.Run("Error in chunking document", func(t *testing.T) {
		doc := golightrag.Document{
			ID:      "test-doc-2",
			Content: "Test content",
		}

		// Create mock handler with error
		handler := &MockDocumentHandler{
			chunkErr: errors.New("chunk error"),
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		err := golightrag.Insert(doc, handler, storage, nil, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error, got nil")
		}

		if storage.kvUpsertSourcesCalled {
			t.Error("Expected KVUpsertSources not to be called")
		}
	})

	t.Run("Error in KV upsert", func(t *testing.T) {
		doc := golightrag.Document{
			ID:      "test-doc-4",
			Content: "Test content",
		}

		// Create mock handler
		handler := &MockDocumentHandler{
			sources: []golightrag.Source{
				{
					Content:    "Test content",
					TokenSize:  2,
					OrderIndex: 0,
				},
			},
		}

		// Create mock storage with error
		storage := &MockStorage{
			kvUpsertSourcesErr: errors.New("kv upsert error"),
		}

		// Call the function under test
		err := golightrag.Insert(doc, handler, storage, nil, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error, got nil")
		}

		if !storage.kvUpsertSourcesCalled {
			t.Error("Expected KVUpsertSources to be called")
		}
	})

	t.Run("Error in entity extraction", func(t *testing.T) {
		doc := golightrag.Document{
			ID:      "test-doc-5",
			Content: "Test content",
		}

		// Create mock LLM with error
		mockLLM := &MockLLM{
			chatErr: errors.New("LLM chat error"),
		}

		// Create mock handler
		handler := &MockDocumentHandler{
			sources: []golightrag.Source{
				{
					Content:    "Test content",
					TokenSize:  2,
					OrderIndex: 0,
				},
			},
			entityExtractionPromptData: golightrag.EntityExtractionPromptData{
				Goal:        "Extract entities",
				EntityTypes: []string{"PERSON", "ORGANIZATION"},
				Language:    "English",
			},
			maxRetries:  0, // Force immediate failure
			gleanCount:  2,
			maxTokenLen: 1000,
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		err := golightrag.Insert(doc, handler, storage, mockLLM, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error, got nil")
		}
	})
}
