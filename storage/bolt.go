package storage

import (
	"fmt"

	golightrag "github.com/lollipopkit/go-light-rag"
	bolt "go.etcd.io/bbolt"
)

// Bolt provides a BoltDB key-value storage implementation of storage interfaces.
// It handles database operations for storing and retrieving source documents.
type Bolt struct {
	DB *bolt.DB
}

// NewBolt creates a new BoltDB client connection with the provided file path.
// It returns an initialized Bolt struct and any error encountered during database setup.
// The function ensures that required buckets exist in the database.
func NewBolt(path string) (Bolt, error) {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return Bolt{}, fmt.Errorf("failed to open bolt database: %w", err)
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte("sources"))
		return err
	}); err != nil {
		return Bolt{}, fmt.Errorf("failed to create sources bucket: %w", err)
	}

	return Bolt{DB: db}, nil
}

// KVSource retrieves a source document by ID from the BoltDB database.
// It returns the found source or an error if the source doesn't exist or if the query fails.
func (b Bolt) KVSource(id string) (golightrag.Source, error) {
	var result golightrag.Source

	err := b.DB.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("sources"))

		content := b.Get([]byte(id))
		if content == nil {
			return fmt.Errorf("source not found")
		}

		result.Content = string(content)

		return nil
	})

	return result, err
}

// KVUpsertSources creates or updates multiple source documents in the BoltDB database.
// It returns an error if any database operation fails during the process.
func (b Bolt) KVUpsertSources(sources []golightrag.Source) error {
	return b.DB.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("sources"))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		for _, chunk := range sources {
			err := b.Put([]byte(chunk.ID), []byte(chunk.Content))
			if err != nil {
				return fmt.Errorf("failed to put sources: %w", err)
			}
		}

		return nil
	})
}
