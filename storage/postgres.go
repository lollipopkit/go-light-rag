package storage

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/pgvector/pgvector-go"
)

const (
	postgresPingTimeout   = 5 * time.Second
	postgresUpsertTimeout = 30 * time.Second
)

// Postgres provides a PostgreSQL implementation covering key-value, vector, and graph storage.
type Postgres struct {
	DB            *sql.DB
	embeddingFunc EmbeddingFunc
	vectorDim     int
	topK          int
}

// NewPostgres creates a PostgreSQL storage with the provided connection string and options.
// vectorDim must match the embedding output dimension when vector search is used.
func NewPostgres(connString string, vectorDim, topK int, embeddingFunc EmbeddingFunc) (Postgres, error) {
	db, err := sql.Open("pgx", connString)
	if err != nil {
		return Postgres{}, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	store, err := newPostgresFromDB(db, vectorDim, topK, embeddingFunc)
	if err != nil {
		_ = db.Close()
		return Postgres{}, err
	}

	return store, nil
}

// NewPostgresWithDB initializes PostgreSQL storage using an existing *sql.DB connection.
// This is useful when the caller manages the database lifecycle or during testing.
func NewPostgresWithDB(db *sql.DB, vectorDim, topK int, embeddingFunc EmbeddingFunc) (Postgres, error) {
	if db == nil {
		return Postgres{}, fmt.Errorf("db is nil")
	}

	return newPostgresFromDB(db, vectorDim, topK, embeddingFunc)
}

func newPostgresFromDB(db *sql.DB, vectorDim, topK int, embeddingFunc EmbeddingFunc) (Postgres, error) {
	if vectorDim <= 0 {
		return Postgres{}, fmt.Errorf("vectorDim must be positive")
	}
	if topK <= 0 {
		topK = 5
	}

	if err := initPostgres(db, vectorDim); err != nil {
		return Postgres{}, err
	}

	return Postgres{DB: db, embeddingFunc: embeddingFunc, vectorDim: vectorDim, topK: topK}, nil
}

func initPostgres(db *sql.DB, vectorDim int) error {
	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping postgres: %w", err)
	}

	statements := []string{
		"CREATE EXTENSION IF NOT EXISTS vector",
		`CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    token_size INTEGER NOT NULL DEFAULT 0,
    order_index INTEGER NOT NULL DEFAULT 0
)`,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    description TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    vector vector(%d) NOT NULL
)`, vectorDim),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS relationships (
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    vector vector(%d) NOT NULL,
    PRIMARY KEY (source_entity, target_entity)
)`, vectorDim),
	}

	for _, stmt := range statements {
		if _, err := db.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to execute init statement: %w", err)
		}
	}

	return nil
}

// KVSource retrieves a source document by ID from the PostgreSQL database.
// It returns the found source or an error if the source doesn't exist or if the query fails.
func (p Postgres) KVSource(id string) (golightrag.Source, error) {
	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	var result golightrag.Source

	row := p.DB.QueryRowContext(ctx, `SELECT id, content, token_size, order_index FROM sources WHERE id = $1`, id)
	if err := row.Scan(&result.ID, &result.Content, &result.TokenSize, &result.OrderIndex); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return golightrag.Source{}, fmt.Errorf("source not found")
		}
		return golightrag.Source{}, fmt.Errorf("failed to get source: %w", err)
	}

	return result, nil
}

// KVUpsertSources creates or updates multiple source documents in the PostgreSQL database.
// It returns an error if any database operation fails during the process.
func (p Postgres) KVUpsertSources(sources []golightrag.Source) error {
	ctx, cancel := context.WithTimeout(context.Background(), postgresUpsertTimeout)
	defer cancel()

	tx, err := p.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx, `
INSERT INTO sources (id, content, token_size, order_index)
VALUES ($1, $2, $3, $4)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    token_size = EXCLUDED.token_size,
    order_index = EXCLUDED.order_index`)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, src := range sources {
		if _, err := stmt.ExecContext(ctx, src.ID, src.Content, src.TokenSize, src.OrderIndex); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("failed to upsert source %s: %w", src.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// VectorQueryEntity performs a semantic search for entities based on the provided keywords.
func (p Postgres) VectorQueryEntity(keywords string) ([]string, error) {
	if p.embeddingFunc == nil {
		return nil, fmt.Errorf("embedding function is not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	vector, err := p.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	rows, err := p.DB.QueryContext(ctx,
		`SELECT entity_id FROM entities ORDER BY vector <=> $1 LIMIT $2`,
		pgvector.NewVector(vector), p.topK)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}
	defer rows.Close()

	results := make([]string, 0, p.topK)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("failed to scan entity_id: %w", err)
		}
		results = append(results, id)
	}

	return results, rows.Err()
}

// VectorQueryRelationship performs a semantic search for relationships based on the provided keywords.
func (p Postgres) VectorQueryRelationship(keywords string) ([][2]string, error) {
	if p.embeddingFunc == nil {
		return nil, fmt.Errorf("embedding function is not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	vector, err := p.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	rows, err := p.DB.QueryContext(ctx,
		`SELECT source_entity, target_entity FROM relationships ORDER BY vector <=> $1 LIMIT $2`,
		pgvector.NewVector(vector), p.topK)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}
	defer rows.Close()

	results := make([][2]string, 0, p.topK)
	for rows.Next() {
		var source, target string
		if err := rows.Scan(&source, &target); err != nil {
			return nil, fmt.Errorf("failed to scan relationship: %w", err)
		}
		results = append(results, [2]string{source, target})
	}

	return results, rows.Err()
}

// VectorUpsertEntity creates or updates an entity vector based on its content.
func (p Postgres) VectorUpsertEntity(name, content string) error {
	if p.embeddingFunc == nil {
		return fmt.Errorf("embedding function is not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresUpsertTimeout)
	defer cancel()

	vector, err := p.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	_, err = p.DB.ExecContext(ctx, `
INSERT INTO entities (entity_id, entity_type, description, source_ids, created_at, vector)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (entity_id) DO UPDATE SET
    entity_type = EXCLUDED.entity_type,
    description = EXCLUDED.description,
    source_ids = EXCLUDED.source_ids,
    created_at = EXCLUDED.created_at,
    vector = EXCLUDED.vector
`, name, "UNKNOWN", content, "", time.Now().UTC(), pgvector.NewVector(vector))
	if err != nil {
		return fmt.Errorf("failed to upsert entity vector: %w", err)
	}

	return nil
}

// VectorUpsertRelationship creates or updates a relationship vector based on its content.
func (p Postgres) VectorUpsertRelationship(source, target, content string) error {
	if p.embeddingFunc == nil {
		return fmt.Errorf("embedding function is not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresUpsertTimeout)
	defer cancel()

	vector, err := p.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	_, err = p.DB.ExecContext(ctx, `
INSERT INTO relationships (source_entity, target_entity, weight, description, keywords, source_ids, created_at, vector)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (source_entity, target_entity) DO UPDATE SET
    weight = EXCLUDED.weight,
    description = EXCLUDED.description,
    keywords = EXCLUDED.keywords,
    source_ids = EXCLUDED.source_ids,
    created_at = EXCLUDED.created_at,
    vector = EXCLUDED.vector
`, source, target, 1.0, content, "", "", time.Now().UTC(), pgvector.NewVector(vector))
	if err != nil {
		return fmt.Errorf("failed to upsert relationship vector: %w", err)
	}

	return nil
}

// GraphEntity retrieves a graph entity by name from the PostgreSQL database.
func (p Postgres) GraphEntity(name string) (golightrag.GraphEntity, error) {
	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	row := p.DB.QueryRowContext(ctx, `
SELECT entity_id, entity_type, description, source_ids, created_at
FROM entities WHERE entity_id = $1`, name)

	var entity golightrag.GraphEntity
	if err := row.Scan(&entity.Name, &entity.Type, &entity.Descriptions, &entity.SourceIDs, &entity.CreatedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return golightrag.GraphEntity{}, golightrag.ErrEntityNotFound
		}
		return golightrag.GraphEntity{}, fmt.Errorf("failed to get entity: %w", err)
	}

	return entity, nil
}

// GraphRelationship retrieves a relationship between two entities.
func (p Postgres) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	row := p.DB.QueryRowContext(ctx, `
SELECT source_entity, target_entity, weight, description, keywords, source_ids, created_at
FROM relationships
WHERE (source_entity = $1 AND target_entity = $2)`, sourceEntity, targetEntity)

	var rel golightrag.GraphRelationship
	var keywords string
	if err := row.Scan(&rel.SourceEntity, &rel.TargetEntity, &rel.Weight, &rel.Descriptions, &keywords, &rel.SourceIDs, &rel.CreatedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return golightrag.GraphRelationship{}, golightrag.ErrRelationshipNotFound
		}
		return golightrag.GraphRelationship{}, fmt.Errorf("failed to get relationship: %w", err)
	}

	if keywords != "" {
		rel.Keywords = strings.Split(keywords, golightrag.GraphFieldSeparator)
	}

	return rel, nil
}

// GraphUpsertEntity creates or updates an entity in PostgreSQL.
func (p Postgres) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	ctx, cancel := context.WithTimeout(context.Background(), postgresUpsertTimeout)
	defer cancel()

	_, err := p.DB.ExecContext(ctx, `
INSERT INTO entities (entity_id, entity_type, description, source_ids, created_at, vector)
VALUES ($1, $2, $3, $4, $5, COALESCE((SELECT vector FROM entities WHERE entity_id = $1), $6))
ON CONFLICT (entity_id) DO UPDATE SET
    entity_type = EXCLUDED.entity_type,
    description = EXCLUDED.description,
    source_ids = EXCLUDED.source_ids,
    created_at = EXCLUDED.created_at
`, entity.Name, entity.Type, entity.Descriptions, entity.SourceIDs, entity.CreatedAt, pgvector.NewVector(make([]float32, p.vectorDim)))
	if err != nil {
		return fmt.Errorf("failed to upsert entity: %w", err)
	}

	return nil
}

// GraphUpsertRelationship creates or updates a relationship in PostgreSQL.
func (p Postgres) GraphUpsertRelationship(relationship golightrag.GraphRelationship) error {
	ctx, cancel := context.WithTimeout(context.Background(), postgresUpsertTimeout)
	defer cancel()

	keywords := strings.Join(relationship.Keywords, golightrag.GraphFieldSeparator)
	_, err := p.DB.ExecContext(ctx, `
INSERT INTO relationships (source_entity, target_entity, weight, description, keywords, source_ids, created_at, vector)
VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE((SELECT vector FROM relationships WHERE source_entity = $1 AND target_entity = $2), $8))
ON CONFLICT (source_entity, target_entity) DO UPDATE SET
    weight = EXCLUDED.weight,
    description = EXCLUDED.description,
    keywords = EXCLUDED.keywords,
    source_ids = EXCLUDED.source_ids,
    created_at = EXCLUDED.created_at
`, relationship.SourceEntity, relationship.TargetEntity, relationship.Weight, relationship.Descriptions, keywords, relationship.SourceIDs, relationship.CreatedAt, pgvector.NewVector(make([]float32, p.vectorDim)))
	if err != nil {
		return fmt.Errorf("failed to upsert relationship: %w", err)
	}

	return nil
}

// GraphEntities batch retrieves multiple entities by name.
func (p Postgres) GraphEntities(names []string) (map[string]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string]golightrag.GraphEntity{}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	placeholders := make([]string, len(names))
	args := make([]any, len(names))
	for i, name := range names {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = name
	}

	query := fmt.Sprintf(`SELECT entity_id, entity_type, description, source_ids, created_at FROM entities WHERE entity_id IN (%s)`, strings.Join(placeholders, ","))

	rows, err := p.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}
	defer rows.Close()

	result := make(map[string]golightrag.GraphEntity)
	for rows.Next() {
		var ent golightrag.GraphEntity
		if err := rows.Scan(&ent.Name, &ent.Type, &ent.Descriptions, &ent.SourceIDs, &ent.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan entity: %w", err)
		}
		result[ent.Name] = ent
	}

	return result, rows.Err()
}

// GraphRelationships batch retrieves relationships by pairs.
func (p Postgres) GraphRelationships(pairs [][2]string) (map[string]golightrag.GraphRelationship, error) {
	if len(pairs) == 0 {
		return map[string]golightrag.GraphRelationship{}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	placeholders := make([]string, len(pairs))
	args := make([]any, 0, len(pairs)*2)
	for i, pair := range pairs {
		placeholders[i] = fmt.Sprintf("($%d, $%d)", i*2+1, i*2+2)
		args = append(args, pair[0], pair[1])
	}

	query := fmt.Sprintf(`SELECT source_entity, target_entity, weight, description, keywords, source_ids, created_at FROM relationships WHERE (source_entity, target_entity) IN (%s)`, strings.Join(placeholders, ","))
	rows, err := p.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}
	defer rows.Close()

	result := make(map[string]golightrag.GraphRelationship)
	for rows.Next() {
		var rel golightrag.GraphRelationship
		var keywords string
		if err := rows.Scan(&rel.SourceEntity, &rel.TargetEntity, &rel.Weight, &rel.Descriptions, &keywords, &rel.SourceIDs, &rel.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan relationship: %w", err)
		}
		if keywords != "" {
			rel.Keywords = strings.Split(keywords, golightrag.GraphFieldSeparator)
		}
		key := fmt.Sprintf("%s-%s", rel.SourceEntity, rel.TargetEntity)
		result[key] = rel
	}

	return result, rows.Err()
}

// GraphCountEntitiesRelationships counts the number of relationships for each entity.
func (p Postgres) GraphCountEntitiesRelationships(names []string) (map[string]int, error) {
	if len(names) == 0 {
		return map[string]int{}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	placeholders := make([]string, len(names))
	args := make([]any, len(names))
	for i, name := range names {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = name
	}

	query := fmt.Sprintf(`
WITH related AS (
    SELECT source_entity AS entity_id FROM relationships WHERE source_entity IN (%s)
    UNION ALL
    SELECT target_entity AS entity_id FROM relationships WHERE target_entity IN (%s)
)
SELECT entity_id, COUNT(*) FROM related GROUP BY entity_id`, strings.Join(placeholders, ","), strings.Join(placeholders, ","))

	rows, err := p.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to count relationships: %w", err)
	}
	defer rows.Close()

	result := make(map[string]int)
	for rows.Next() {
		var name string
		var count int
		if err := rows.Scan(&name, &count); err != nil {
			return nil, fmt.Errorf("failed to scan count: %w", err)
		}
		result[name] = count
	}

	return result, rows.Err()
}

// GraphRelatedEntities finds entities directly connected to specified names.
func (p Postgres) GraphRelatedEntities(names []string) (map[string][]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string][]golightrag.GraphEntity{}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), postgresPingTimeout)
	defer cancel()

	results := make(map[string][]golightrag.GraphEntity)
	for _, name := range names {
		rows, err := p.DB.QueryContext(ctx, `
SELECT DISTINCT CASE WHEN source_entity = $1 THEN target_entity ELSE source_entity END AS related
FROM relationships
WHERE source_entity = $1 OR target_entity = $1`, name)
		if err != nil {
			return nil, fmt.Errorf("failed to query related entities: %w", err)
		}

		relatedIDs := make([]string, 0)
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, fmt.Errorf("failed to scan related entity: %w", err)
			}
			relatedIDs = append(relatedIDs, id)
		}
		relErr := rows.Err()
		rows.Close()
		if relErr != nil {
			return nil, relErr
		}

		ents, err := p.GraphEntities(relatedIDs)
		if err != nil {
			return nil, err
		}

		arr := make([]golightrag.GraphEntity, 0, len(ents))
		for _, ent := range ents {
			arr = append(arr, ent)
		}
		results[name] = arr
	}

	return results, nil
}
