package storage

import (
	"context"
	"errors"
	"fmt"
	"time"

	golightrag "github.com/lollipopkit/go-light-rag"
	"github.com/redis/go-redis/v9"
)

// Redis provides a Redis key-value storage implementation of storage interfaces.
// It handles database operations for storing and retrieving source documents.
type Redis struct {
	Client *redis.Client
}

// NewRedis creates a new Redis client connection with the provided configuration.
// It returns an initialized Redis struct and any error encountered during connection setup.
func NewRedis(addr, password string, db int) (Redis, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := client.Ping(ctx).Result()
	if err != nil {
		return Redis{}, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return Redis{
		Client: client,
	}, nil
}

// KVSource retrieves a source document by ID from the Redis database.
// It returns the found source or an error if the source doesn't exist or if the query fails.
func (r Redis) KVSource(id string) (golightrag.Source, error) {
	var result golightrag.Source

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	content, err := r.Client.Get(ctx, id).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return result, fmt.Errorf("source not found")
		}
		return result, fmt.Errorf("failed to get source: %w", err)
	}

	result.Content = content

	return result, nil
}

// KVUpsertSources creates or updates multiple source documents in the Redis database.
// It returns an error if any database operation fails during the process.
func (r Redis) KVUpsertSources(sources []golightrag.Source) error {
	pipe := r.Client.Pipeline()

	setCtx, setCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer setCancel()

	for _, source := range sources {
		pipe.Set(setCtx, source.ID, source.Content, 0)
	}

	execCtx, execCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer execCancel()

	_, err := pipe.Exec(execCtx)
	if err != nil {
		return fmt.Errorf("failed to execute pipeline: %w", err)
	}

	return nil
}
