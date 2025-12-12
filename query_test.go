package golightrag_test

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
)

func TestQuery(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	// logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
	// 	Level: slog.LevelDebug,
	// }))

	t.Run("Successful query", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM that returns valid keyword extraction
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Entity1", "Knowledge"},
			"low_level_keywords":  {"Entity1", "Information"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
			chatCalls:    make([][]string, 0),
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
		}

		// Create mock storage with predefined data
		storage := &MockStorage{
			// Entity data
			entities: map[string]golightrag.GraphEntity{
				"ENTITY1": {
					Name:         "ENTITY1",
					Type:         "PERSON",
					Descriptions: "Description of Entity1",
					SourceIDs:    "doc-1-chunk-0",
				},
				"ENTITY2": {
					Name:         "ENTITY2",
					Type:         "ORGANIZATION",
					Descriptions: "Description of Entity2",
					SourceIDs:    "doc-1-chunk-0",
				},
			},
			// Relationship data
			relationships: map[string]golightrag.GraphRelationship{
				"ENTITY1:ENTITY2": {
					SourceEntity: "ENTITY1",
					TargetEntity: "ENTITY2",
					Descriptions: "Entity1 is related to Entity2",
					Keywords:     []string{"RELATED_TO", "RELATED", "TO"},
					Weight:       1.0,
					SourceIDs:    "doc-1-chunk-0",
				},
			},
			// Vector query results
			vectorQueryEntityResults: []string{"ENTITY1"},
			vectorQueryRelationshipResults: [][2]string{
				{"ENTITY1", "ENTITY2"},
			},
			// For related entities lookup
			entityRelatedEntitiesMap: map[string][]golightrag.GraphEntity{
				"ENTITY1": {
					{
						Name:         "ENTITY2",
						Type:         "ORGANIZATION",
						Descriptions: "Description of Entity2",
						SourceIDs:    "doc-1-chunk-0",
					},
				},
			},
			// For entity relationship counts
			entityRelationshipCountMap: map[string]int{
				"ENTITY1": 1,
				"ENTITY2": 1,
			},
			// Source data
			sources: map[string]golightrag.Source{
				"doc-1-chunk-0": {
					ID:         "doc-1-chunk-0",
					Content:    "Content about Entity1 and Entity2",
					TokenSize:  10,
					OrderIndex: 0,
				},
			},
		}

		// Call the function under test
		result, err := golightrag.Query(conversations, handler, storage, mockLLM, logger)
		// Assertions
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		// Check that the result contains entities and relationships
		if len(result.LocalEntities) == 0 && len(result.GlobalEntities) == 0 {
			t.Error("Expected entities in result, got none")
		}

		if len(result.LocalRelationships) == 0 && len(result.GlobalRelationships) == 0 {
			t.Error("Expected relationships in result, got none")
		}

		// Check chat was called with expected prompt
		if len(mockLLM.chatCalls) == 0 {
			t.Error("Expected LLM to be called")
		}
	})

	t.Run("Error in extracting query", func(t *testing.T) {
		// Create conversations with no user message
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleAssistant,
				Message: "I am an assistant",
			},
		}

		// Create mock handler and storage
		handler := &MockQueryHandler{}
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, nil, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to missing user message, got nil")
		}
	})

	t.Run("Error in LLM chat", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with error
		mockLLM := &MockLLM{
			chatErr: errors.New("LLM chat error"),
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, mockLLM, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to LLM chat error, got nil")
		}
	})

	t.Run("Error in JSON unmarshaling", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with invalid JSON
		mockLLM := &MockLLM{
			chatResponse: "this is not valid JSON",
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, mockLLM, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to invalid JSON, got nil")
		}
	})

	t.Run("Error in vector query entity", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with valid response
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Entity1"},
			"low_level_keywords":  {"Entity1"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
		}

		// Create mock storage with vector query error
		storage := &MockStorage{
			vectorQueryEntityErr: errors.New("vector query entity error"),
		}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, mockLLM, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to vector query entity error, got nil")
		}
	})

	t.Run("Empty query results", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Unknown",
			},
		}

		// Create mock LLM with valid response
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Unknown"},
			"low_level_keywords":  {"Unknown"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
		}

		// Create mock storage with empty results
		storage := &MockStorage{
			vectorQueryEntityResults:       []string{},
			vectorQueryRelationshipResults: [][2]string{},
		}

		// Call the function under test
		result, err := golightrag.Query(conversations, handler, storage, mockLLM, logger)
		// Assertions
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		// Check that results are empty
		if len(result.LocalEntities) != 0 {
			t.Errorf("Expected 0 local entities, got %d", len(result.LocalEntities))
		}
		if len(result.GlobalEntities) != 0 {
			t.Errorf("Expected 0 global entities, got %d", len(result.GlobalEntities))
		}
	})
}

//nolint:gocognit // Test function with multiple complex test cases and data structures
func TestQueryResultString(t *testing.T) {
	t.Run("Result with sorting by reference count", func(t *testing.T) {
		// Create a sample QueryResult with items having different reference counts
		result := golightrag.QueryResult{
			GlobalEntities: []golightrag.EntityContext{
				{
					Name:        "GlobalEntityLow",
					Type:        "PERSON",
					Description: "Low ref count entity",
					RefCount:    1,
					CreatedAt:   time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			LocalEntities: []golightrag.EntityContext{
				{
					Name:        "LocalEntityHigh",
					Type:        "ORGANIZATION",
					Description: "High ref count entity",
					RefCount:    5,
					CreatedAt:   time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
			GlobalRelationships: []golightrag.RelationshipContext{
				{
					Source:      "SourceLow",
					Target:      "TargetLow",
					Keywords:    "low_ref",
					Description: "Low ref count relationship",
					Weight:      0.5,
					RefCount:    2,
					CreatedAt:   time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
				},
			},
			LocalRelationships: []golightrag.RelationshipContext{
				{
					Source:      "SourceHigh",
					Target:      "TargetHigh",
					Keywords:    "high_ref",
					Description: "High ref count relationship",
					Weight:      0.8,
					RefCount:    7,
					CreatedAt:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
				},
			},
			GlobalSources: []golightrag.SourceContext{
				{
					Content:  "Low ref source",
					RefCount: 3,
				},
			},
			LocalSources: []golightrag.SourceContext{
				{
					Content:  "High ref source",
					RefCount: 10,
				},
			},
		}

		// Call the String method
		output := result.String()

		// Check that the output has the three sections
		if !strings.Contains(output, "-----Entities-----") {
			t.Error("Output missing Entities section")
		}
		if !strings.Contains(output, "-----Relationships-----") {
			t.Error("Output missing Relationships section")
		}
		if !strings.Contains(output, "-----Sources-----") {
			t.Error("Output missing Sources section")
		}

		// Check that the output contains CSV content wrapped in backticks
		if !strings.Contains(output, "```csv") {
			t.Error("Output missing CSV format markers")
		}

		// Check that all entities, relationships, and sources are present
		expectedStrings := []string{
			"LocalEntityHigh", "ORGANIZATION", "High ref count entity",
			"GlobalEntityLow", "PERSON", "Low ref count entity",
			"SourceHigh", "TargetHigh", "high_ref", "High ref count relationship",
			"SourceLow", "TargetLow", "low_ref", "Low ref count relationship",
			"High ref source", "Low ref source",
		}

		for _, str := range expectedStrings {
			if !strings.Contains(output, str) {
				t.Errorf("Output missing expected content: %s", str)
			}
		}

		// Check ordering by reference count (higher ref count should come first)
		highEntityIdx := strings.Index(output, "LocalEntityHigh")
		lowEntityIdx := strings.Index(output, "GlobalEntityLow")
		if highEntityIdx > lowEntityIdx && highEntityIdx != -1 && lowEntityIdx != -1 {
			t.Error("Entity with higher ref count should come before entity with lower ref count")
		}

		highRelIdx := strings.Index(output, "SourceHigh")
		lowRelIdx := strings.Index(output, "SourceLow")
		if highRelIdx > lowRelIdx && highRelIdx != -1 && lowRelIdx != -1 {
			t.Error("Relationship with higher ref count should come before relationship with lower ref count")
		}

		highSourceIdx := strings.Index(output, "High ref source")
		lowSourceIdx := strings.Index(output, "Low ref source")
		if highSourceIdx > lowSourceIdx && highSourceIdx != -1 && lowSourceIdx != -1 {
			t.Error("Source with higher ref count should come before source with lower ref count")
		}
	})

	t.Run("Empty result", func(t *testing.T) {
		// Create an empty QueryResult
		result := golightrag.QueryResult{}

		// Call the String method
		output := result.String()

		// Verify the output has the section headers
		if !strings.Contains(output, "-----Entities-----") {
			t.Error("Output missing Entities section")
		}
		if !strings.Contains(output, "-----Relationships-----") {
			t.Error("Output missing Relationships section")
		}
		if !strings.Contains(output, "-----Sources-----") {
			t.Error("Output missing Sources section")
		}

		// The output should still have CSV headers even with empty data
		if !strings.Contains(output, "id,name,type,description,ref_count,created_at") {
			t.Error("Output missing entity CSV header")
		}
		if !strings.Contains(output, "id,source,target,keywords,description,weight,ref_count,created_at") {
			t.Error("Output missing relationship CSV header")
		}
		if !strings.Contains(output, "id,content,ref_count") {
			t.Error("Output missing source CSV header")
		}
	})
}
