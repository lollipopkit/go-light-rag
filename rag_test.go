package golightrag_test

import (
	"errors"
	"fmt"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
)

type MockDocumentHandler struct {
	chunkErr                   error
	sources                    []golightrag.Source
	entityExtractionPromptData golightrag.EntityExtractionPromptData

	maxRetries       int
	concurrencyCount int
	backoffDuration  time.Duration
	gleanCount       int
	maxTokenLen      int
}

type MockQueryHandler struct {
	keywordExtractionPromptData golightrag.KeywordExtractionPromptData
}

type MockLLM struct {
	chatResponse string
	chatErr      error

	// For tracking interactions
	chatCalls [][]string
}

type MockStorage struct {
	kvUpsertSourcesErr          error
	graphEntityErr              error
	graphRelationshipErr        error
	graphUpsertEntityErr        error
	vectorUpsertEntityErr       error
	graphUpsertRelationshipErr  error
	vectorUpsertRelationshipErr error

	// Track calls to methods
	kvUpsertSourcesCalled          bool
	graphEntityCalled              bool
	graphRelationshipCalled        bool
	graphUpsertEntityCalled        bool
	graphUpsertRelationshipCalled  bool
	vectorUpsertEntityCalled       bool
	vectorUpsertRelationshipCalled bool

	// Track entities and relationships
	entities      map[string]golightrag.GraphEntity
	relationships map[string]golightrag.GraphRelationship

	sources                        map[string]golightrag.Source
	vectorQueryEntityResults       []string
	vectorQueryRelationshipResults [][2]string
	entityRelatedEntitiesMap       map[string][]golightrag.GraphEntity
	entityRelationshipCountMap     map[string]int

	vectorQueryEntityErr       error
	vectorQueryRelationshipErr error
}

func (m *MockDocumentHandler) ChunksDocument(string) ([]golightrag.Source, error) {
	if m.chunkErr != nil {
		return nil, m.chunkErr
	}
	return m.sources, nil
}

func (m *MockDocumentHandler) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	return m.entityExtractionPromptData
}

func (m *MockDocumentHandler) MaxRetries() int {
	return m.maxRetries
}

func (m *MockDocumentHandler) BackoffDuration() time.Duration {
	return m.backoffDuration
}

func (m *MockDocumentHandler) ConcurrencyCount() int {
	return m.concurrencyCount
}

func (m *MockDocumentHandler) GleanCount() int {
	return m.gleanCount
}

func (m *MockDocumentHandler) MaxSummariesTokenLength() int {
	return m.maxTokenLen
}

func (m *MockQueryHandler) KeywordExtractionPromptData() golightrag.KeywordExtractionPromptData {
	return m.keywordExtractionPromptData
}

func (m *MockLLM) Chat(messages []string) (string, error) {
	// Record this call
	if m.chatCalls != nil {
		m.chatCalls = append(m.chatCalls, messages)
	}

	if m.chatErr != nil {
		return "", m.chatErr
	}
	return m.chatResponse, nil
}

func (m *MockStorage) KVSource(id string) (golightrag.Source, error) {
	if source, ok := m.sources[id]; ok {
		return source, nil
	}
	return golightrag.Source{}, errors.New("source not found")
}

func (m *MockStorage) KVUpsertSources([]golightrag.Source) error {
	m.kvUpsertSourcesCalled = true
	return m.kvUpsertSourcesErr
}

func (m *MockStorage) GraphEntity(name string) (golightrag.GraphEntity, error) {
	m.graphEntityCalled = true
	if m.graphEntityErr != nil {
		return golightrag.GraphEntity{}, m.graphEntityErr
	}

	// Check if the entity exists in our tracked map
	if entity, exists := m.entities[name]; exists {
		return entity, nil
	}

	return golightrag.GraphEntity{}, golightrag.ErrEntityNotFound
}

func (m *MockStorage) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	m.graphRelationshipCalled = true
	if m.graphRelationshipErr != nil {
		return golightrag.GraphRelationship{}, m.graphRelationshipErr
	}

	// Check if the relationship exists in our tracked map
	key := fmt.Sprintf("%s:%s", sourceEntity, targetEntity)
	if rel, exists := m.relationships[key]; exists {
		return rel, nil
	}

	return golightrag.GraphRelationship{}, golightrag.ErrRelationshipNotFound
}

func (m *MockStorage) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	m.graphUpsertEntityCalled = true
	if m.graphUpsertEntityErr != nil {
		return m.graphUpsertEntityErr
	}

	if m.entities == nil {
		m.entities = make(map[string]golightrag.GraphEntity)
	}

	// Store the entity
	m.entities[entity.Name] = entity
	return nil
}

func (m *MockStorage) GraphEntities(names []string) (map[string]golightrag.GraphEntity, error) {
	result := make(map[string]golightrag.GraphEntity)
	for _, name := range names {
		if entity, exists := m.entities[name]; exists {
			result[name] = entity
		}
	}
	return result, nil
}

func (m *MockStorage) GraphUpsertRelationship(relationship golightrag.GraphRelationship) error {
	m.graphUpsertRelationshipCalled = true
	if m.graphUpsertRelationshipErr != nil {
		return m.graphUpsertRelationshipErr
	}

	if m.relationships == nil {
		m.relationships = make(map[string]golightrag.GraphRelationship)
	}

	// Store the relationship
	key := fmt.Sprintf("%s:%s", relationship.SourceEntity, relationship.TargetEntity)
	m.relationships[key] = relationship
	return nil
}

func (m *MockStorage) GraphRelationships(pairs [][2]string) (map[string]golightrag.GraphRelationship, error) {
	result := make(map[string]golightrag.GraphRelationship)
	for _, pair := range pairs {
		if len(pair) != 2 {
			continue
		}
		sourceEntity := pair[0]
		targetEntity := pair[1]

		// Look for the relationship in our mock data
		lookupKey := fmt.Sprintf("%s:%s", sourceEntity, targetEntity)

		if rel, exists := m.relationships[lookupKey]; exists {
			// Store with the expected key format
			key := fmt.Sprintf("%s-%s", sourceEntity, targetEntity)
			result[key] = rel
		}
	}
	return result, nil
}

func (m *MockStorage) GraphCountEntitiesRelationships(names []string) (map[string]int, error) {
	result := make(map[string]int)
	for _, name := range names {
		if count, ok := m.entityRelationshipCountMap[name]; ok {
			result[name] = count
		} else {
			result[name] = 0
		}
	}
	return result, nil
}

func (m *MockStorage) GraphRelatedEntities(names []string) (map[string][]golightrag.GraphEntity, error) {
	result := make(map[string][]golightrag.GraphEntity)
	for _, name := range names {
		if entities, ok := m.entityRelatedEntitiesMap[name]; ok {
			result[name] = entities
		} else {
			result[name] = []golightrag.GraphEntity{}
		}
	}
	return result, nil
}

func (m *MockStorage) VectorUpsertEntity(_, _ string) error {
	m.vectorUpsertEntityCalled = true
	return m.vectorUpsertEntityErr
}

func (m *MockStorage) VectorUpsertRelationship(_, _, _ string) error {
	m.vectorUpsertRelationshipCalled = true
	return m.vectorUpsertRelationshipErr
}

func (m *MockStorage) VectorQueryEntity(string) ([]string, error) {
	if m.vectorQueryEntityErr != nil {
		return nil, m.vectorQueryEntityErr
	}
	return m.vectorQueryEntityResults, nil
}

func (m *MockStorage) VectorQueryRelationship(string) ([][2]string, error) {
	if m.vectorQueryRelationshipErr != nil {
		return nil, m.vectorQueryRelationshipErr
	}
	return m.vectorQueryRelationshipResults, nil
}
