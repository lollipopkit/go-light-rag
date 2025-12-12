package storage

import (
	"context"
	"fmt"
	"strings"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j/db"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j/dbtype"
)

// Neo4J provides a Neo4j graph database implementation of storage interfaces.
// It handles database connections and operations for storing and retrieving graph entities
// and relationships.
type Neo4J struct {
	Client neo4j.DriverWithContext
}

// NewNeo4J creates a new Neo4j client connection with the provided connection parameters.
// It returns an initialized Neo4J struct and any error encountered during connection setup.
// The returned Neo4J instance must be closed with Close() when no longer needed to free up resources.
func NewNeo4J(target, user, password string) (Neo4J, error) {
	driver, err := neo4j.NewDriverWithContext(
		target,
		neo4j.BasicAuth(user, password, ""))
	if err != nil {
		return Neo4J{}, fmt.Errorf("failed to create neo4j driver: %w", err)
	}
	return Neo4J{Client: driver}, nil
}

func graphEntityFromNode(node dbtype.Node) golightrag.GraphEntity {
	name, ok := node.Props["entity_id"].(string)
	if !ok {
		name = ""
	}
	typ, ok := node.Props["entity_type"].(string)
	if !ok {
		typ = ""
	}
	desc, ok := node.Props["description"].(string)
	if !ok {
		desc = ""
	}
	sourceIDs, ok := node.Props["source_ids"].(string)
	if !ok {
		sourceIDs = ""
	}
	createdAtStr, ok := node.Props["created_at"].(string)
	if !ok {
		createdAtStr = time.Now().Format(time.RFC3339)
	}
	createdAt, err := time.Parse(time.RFC3339, createdAtStr)
	if err != nil {
		createdAt = time.Now()
	}

	return golightrag.GraphEntity{
		Name:         name,
		Type:         typ,
		Descriptions: desc,
		SourceIDs:    sourceIDs,
		CreatedAt:    createdAt,
	}
}

func graphRelationshipFromEdge(source, target string, props map[string]any) golightrag.GraphRelationship {
	weight, ok := props["weight"].(float64)
	if !ok {
		weight = 1.0
	}
	description, ok := props["description"].(string)
	if !ok {
		description = ""
	}
	keywords, ok := props["keywords"].(string)
	if !ok {
		keywords = ""
	}
	arrKeywords := strings.Split(keywords, golightrag.GraphFieldSeparator)
	sourceIDs, ok := props["source_ids"].(string)
	if !ok {
		sourceIDs = ""
	}
	createdAtStr, ok := props["created_at"].(string)
	if !ok {
		createdAtStr = time.Now().Format(time.RFC3339)
	}
	createdAt, err := time.Parse(time.RFC3339, createdAtStr)
	if err != nil {
		createdAt = time.Now()
	}

	return golightrag.GraphRelationship{
		SourceEntity: source,
		TargetEntity: target,
		Weight:       weight,
		Descriptions: description,
		Keywords:     arrKeywords,
		SourceIDs:    sourceIDs,
		CreatedAt:    createdAt,
	}
}

// GraphEntity retrieves a graph entity by name from the Neo4j database.
// It returns the found entity or an error if the entity doesn't exist or if the query fails.
func (n Neo4J) GraphEntity(name string) (golightrag.GraphEntity, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := "MATCH (n:base {entity_id: $entityID}) RETURN n"
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entityID": name,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			ety, err := queryRes.Single(ctx)
			if err != nil {
				return nil, golightrag.ErrEntityNotFound
			}
			return ety, nil
		})
	})
	if err != nil {
		return golightrag.GraphEntity{}, err
	}
	record, ok := res.(*db.Record)
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("invalid result type, got %T, want *db.Record", res)
	}
	nNode, ok := record.Get("n")
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("expected n key is not found")
	}
	node, ok := nNode.(dbtype.Node)
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("invalid n type, got %T, want dbtype.Node", n)
	}

	return graphEntityFromNode(node), nil
}

// GraphRelationship retrieves a relationship between two entities from the Neo4j database.
// It returns the found relationship or an error if the relationship doesn't exist or if the query fails.
func (n Neo4J) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
RETURN properties(r) as edge_properties
      `
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"source_entity_id": sourceEntity,
				"target_entity_id": targetEntity,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			ety, err := queryRes.Single(ctx)
			if err != nil {
				return nil, golightrag.ErrRelationshipNotFound
			}
			return ety, nil
		})
	})
	if err != nil {
		return golightrag.GraphRelationship{}, err
	}
	record, ok := res.(*db.Record)
	if !ok {
		return golightrag.GraphRelationship{}, fmt.Errorf("invalid result type, got %T, want *db.Record", res)
	}
	edgeProps, ok := record.Get("edge_properties")
	if !ok {
		return golightrag.GraphRelationship{}, fmt.Errorf("expected edge_properties key is not found")
	}
	props, ok := edgeProps.(map[string]any)
	if !ok {
		return golightrag.GraphRelationship{},
			fmt.Errorf("invalid edge_properties type, got %T, want map[string]any", edgeProps)
	}

	return graphRelationshipFromEdge(sourceEntity, targetEntity, props), nil
}

// GraphUpsertEntity creates or updates an entity in the Neo4j graph database.
// It returns an error if the database operation fails.
func (n Neo4J) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	_, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			return tx.Run(
				ctx,
				fmt.Sprintf(`
MERGE (n:base {entity_id: $properties.entity_id})
SET n += $properties
SET n:%s`, "`"+entity.Type+"`"),
				map[string]any{
					"properties": map[string]any{
						"entity_id":   entity.Name,
						"entity_type": entity.Type,
						"description": entity.Descriptions,
						"source_ids":  entity.SourceIDs,
						"created_at":  entity.CreatedAt.Format(time.RFC3339),
					},
				},
			)
		})
	})

	return err
}

// GraphUpsertRelationship creates or updates a relationship between two entities in the Neo4j graph database.
// It returns an error if the database operation fails.
func (n Neo4J) GraphUpsertRelationship(relationship golightrag.GraphRelationship) error {
	_, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			keywords := strings.Join(relationship.Keywords, golightrag.GraphFieldSeparator)
			return tx.Run(
				ctx,
				`
MATCH (source:base {entity_id: $source_entity_id})
WITH source
MATCH (target:base {entity_id: $target_entity_id})
MERGE (source)-[r:DIRECTED]-(target)
SET r += $properties
`,
				map[string]any{
					"source_entity_id": relationship.SourceEntity,
					"target_entity_id": relationship.TargetEntity,
					"properties": map[string]any{
						"weight":      relationship.Weight,
						"description": relationship.Descriptions,
						"keywords":    keywords,
						"source_ids":  relationship.SourceIDs,
						"created_at":  relationship.CreatedAt.Format(time.RFC3339),
					},
				},
			)
		})
	})

	return err
}

// GraphEntities retrieves multiple graph entities by their names from the Neo4j database.
// It returns a map of entity names to GraphEntity objects, or an error if the query fails.
func (n Neo4J) GraphEntities(names []string) (map[string]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string]golightrag.GraphEntity{}, nil
	}

	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (n:base) 
WHERE n.entity_id IN $entityIDs 
RETURN n, n.entity_id as entity_id`
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entityIDs": names,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			result := make(map[string]dbtype.Node)
			for record, err := range queryRes.Records(ctx) {
				if err != nil {
					return nil, fmt.Errorf("failed to get result: %w", err)
				}

				node, ok := record.Get("n")
				if !ok {
					continue
				}

				entityID, ok := record.Get("entity_id")
				if !ok {
					continue
				}

				entityIDStr, ok := entityID.(string)
				if !ok {
					continue
				}

				dbNode, ok := node.(dbtype.Node)
				if !ok {
					continue
				}

				result[entityIDStr] = dbNode
			}

			return result, nil
		})
	})
	if err != nil {
		return nil, err
	}

	nodeMap, ok := res.(map[string]dbtype.Node)
	if !ok {
		return nil, fmt.Errorf("invalid result type, got %T, want map[string]dbtype.Node", res)
	}

	entities := make(map[string]golightrag.GraphEntity)
	for name, node := range nodeMap {
		entities[name] = graphEntityFromNode(node)
	}

	return entities, nil
}

// GraphRelationships retrieves multiple relationships between entity pairs from the Neo4j database.
// It returns a map where the key is "sourceEntity-targetEntity" and the value is the GraphRelationship.
func (n Neo4J) GraphRelationships(pairs [][2]string) (map[string]golightrag.GraphRelationship, error) {
	if len(pairs) == 0 {
		return map[string]golightrag.GraphRelationship{}, nil
	}

	// Prepare parameters for the query
	sources := make([]string, len(pairs))
	targets := make([]string, len(pairs))
	for i, pair := range pairs {
		sources[i] = pair[0]
		targets[i] = pair[1]
	}

	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
UNWIND $pairs AS pair
MATCH (start:base {entity_id: pair[0]})-[r]-(end:base {entity_id: pair[1]})
RETURN pair[0] as source, pair[1] as target, properties(r) as edge_properties
			`

			// Convert pairs to a format suitable for the query
			pairsParam := make([][]string, len(pairs))
			for i, pair := range pairs {
				pairsParam[i] = []string{pair[0], pair[1]}
			}

			queryRes, err := tx.Run(ctx, query, map[string]any{
				"pairs": pairsParam,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			result := make(map[string]map[string]any)
			for record, err := range queryRes.Records(ctx) {
				if err != nil {
					return nil, fmt.Errorf("failed to get result: %w", err)
				}

				source, sourceOK := record.Get("source")
				target, targetOK := record.Get("target")
				edgeProps, propsOK := record.Get("edge_properties")

				if !sourceOK || !targetOK || !propsOK {
					continue
				}

				sourceStr, sourceOK := source.(string)
				targetStr, targetOK := target.(string)
				props, propsOK := edgeProps.(map[string]any)

				if !sourceOK || !targetOK || !propsOK {
					continue
				}

				key := fmt.Sprintf("%s-%s", sourceStr, targetStr)
				result[key] = props
			}

			return result, nil
		})
	})
	if err != nil {
		return nil, err
	}

	propsMap, ok := res.(map[string]map[string]any)
	if !ok {
		return nil, fmt.Errorf("invalid result type, got %T, want map[string]map[string]any", res)
	}

	relationships := make(map[string]golightrag.GraphRelationship)
	for key, props := range propsMap {
		parts := strings.Split(key, "-")
		if len(parts) != 2 {
			continue
		}

		rel := graphRelationshipFromEdge(parts[0], parts[1], props)
		relationships[key] = rel
	}

	return relationships, nil
}

// GraphCountEntitiesRelationships counts the number of relationships for multiple entities.
// It returns a map of entity names to their relationship counts.
func (n Neo4J) GraphCountEntitiesRelationships(names []string) (map[string]int, error) {
	if len(names) == 0 {
		return map[string]int{}, nil
	}

	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (n:base)
WHERE n.entity_id IN $entity_ids
OPTIONAL MATCH (n)-[r]-()
RETURN n.entity_id AS entity_id, COUNT(r) AS degree
            `
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entity_ids": names,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			result := make(map[string]int64)
			for record, err := range queryRes.Records(ctx) {
				if err != nil {
					return nil, fmt.Errorf("failed to get result: %w", err)
				}

				entityID, idOK := record.Get("entity_id")
				degree, degreeOK := record.Get("degree")

				if !idOK || !degreeOK {
					continue
				}

				entityIDStr, idOK := entityID.(string)
				degreeCnt, degreeOK := degree.(int64)

				if !idOK || !degreeOK {
					continue
				}

				result[entityIDStr] = degreeCnt
			}

			return result, nil
		})
	})
	if err != nil {
		return nil, err
	}

	countMap, ok := res.(map[string]int64)
	if !ok {
		return nil, fmt.Errorf("invalid result type, got %T, want map[string]int64", res)
	}

	// Convert int64 to int
	counts := make(map[string]int)
	for name, count := range countMap {
		counts[name] = int(count)
	}

	return counts, nil
}

// GraphRelatedEntities retrieves all entities related to multiple input entities.
// It returns a map of entity names to slices of related GraphEntity objects.
//
//nolint:gocognit
func (n Neo4J) GraphRelatedEntities(names []string) (map[string][]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string][]golightrag.GraphEntity{}, nil
	}

	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (n:base)
WHERE n.entity_id IN $entity_ids
OPTIONAL MATCH (n)-[r]-(connected:base)
WHERE connected.entity_id IS NOT NULL
RETURN n.entity_id as source_id, collect(connected) as connected_nodes
            `
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entity_ids": names,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			result := make(map[string][]dbtype.Node)
			for record, err := range queryRes.Records(ctx) {
				if err != nil {
					return nil, fmt.Errorf("failed to get result: %w", err)
				}

				sourceID, sourceOK := record.Get("source_id")
				connectedNodes, connectedOK := record.Get("connected_nodes")

				if !sourceOK || !connectedOK {
					continue
				}

				sourceIDStr, sourceOK := sourceID.(string)
				nodes, connectedOK := connectedNodes.([]any)

				if !sourceOK || !connectedOK {
					continue
				}

				nodeList := make([]dbtype.Node, 0, len(nodes))
				for _, node := range nodes {
					if dbNode, ok := node.(dbtype.Node); ok {
						nodeList = append(nodeList, dbNode)
					}
				}

				result[sourceIDStr] = nodeList
			}

			return result, nil
		})
	})
	if err != nil {
		return nil, err
	}

	nodesMap, ok := res.(map[string][]dbtype.Node)
	if !ok {
		return nil, fmt.Errorf("invalid result type, got %T, want map[string][]dbtype.Node", res)
	}

	relatedEntities := make(map[string][]golightrag.GraphEntity, len(nodesMap))
	for name, nodes := range nodesMap {
		entities := make([]golightrag.GraphEntity, 0, len(nodes))
		for _, node := range nodes {
			entities = append(entities, graphEntityFromNode(node))
		}
		relatedEntities[name] = entities
	}

	return relatedEntities, nil
}

// Close terminates the connection to the Neo4j database.
// It returns any error encountered during the closing operation.
func (n Neo4J) Close(ctx context.Context) error {
	return n.Client.Close(ctx)
}

func (n Neo4J) session(sessFunc func(context.Context, neo4j.SessionWithContext) (any, error)) (any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	sess := n.Client.NewSession(ctx, neo4j.SessionConfig{})
	defer func() {
		closeCtx, closeCancel := context.WithTimeout(context.Background(), time.Second*30)
		defer closeCancel()
		_ = sess.Close(closeCtx)
	}()

	trxCtx, trxCancel := context.WithTimeout(context.Background(), time.Second*30)
	defer trxCancel()

	return sessFunc(trxCtx, sess)
}
