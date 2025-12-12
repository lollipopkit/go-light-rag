package storage

import (
	"regexp"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	golightrag "github.com/lollipopkit/go-light-rag"
)

func TestNewPostgresWithDB(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectPing()
	mock.ExpectExec("CREATE EXTENSION IF NOT EXISTS vector").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS sources").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS entities").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS relationships").WillReturnResult(sqlmock.NewResult(0, 0))

	if _, err := NewPostgresWithDB(db, 3, 5, nil); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestPostgresKVUpsertSources(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectPing()
	mock.ExpectExec("CREATE EXTENSION IF NOT EXISTS vector").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS sources").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS entities").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS relationships").WillReturnResult(sqlmock.NewResult(0, 0))

	store, err := NewPostgresWithDB(db, 3, 5, nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	insertStmt := `INSERT INTO sources (id, content, token_size, order_index)
VALUES ($1, $2, $3, $4)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    token_size = EXCLUDED.token_size,
    order_index = EXCLUDED.order_index`

	mock.ExpectBegin()
	prep := mock.ExpectPrepare(regexp.QuoteMeta(insertStmt))
	prep.ExpectExec().WithArgs("id-1", "content", 3, 1).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectCommit()

	if err := store.KVUpsertSources([]golightrag.Source{{ID: "id-1", Content: "content", TokenSize: 3, OrderIndex: 1}}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestPostgresKVSource(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectPing()
	mock.ExpectExec("CREATE EXTENSION IF NOT EXISTS vector").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS sources").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS entities").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS relationships").WillReturnResult(sqlmock.NewResult(0, 0))

	store, err := NewPostgresWithDB(db, 3, 5, nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	mock.ExpectQuery(regexp.QuoteMeta("SELECT id, content, token_size, order_index FROM sources WHERE id = $1")).
		WithArgs("id-1").
		WillReturnRows(sqlmock.NewRows([]string{"id", "content", "token_size", "order_index"}).AddRow("id-1", "content", 2, 0))

	src, err := store.KVSource("id-1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if src.ID != "id-1" || src.Content != "content" || src.TokenSize != 2 || src.OrderIndex != 0 {
		t.Fatalf("unexpected source returned: %+v", src)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}
