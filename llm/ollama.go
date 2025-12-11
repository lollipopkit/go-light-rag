package llm

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Ollama provides an implementation of the LLM interface for interacting with Ollama's language models.
// It manages connections to an Ollama server instance and handles streaming chat completions.
type Ollama struct {
	host  string
	model string

	params Parameters

	client *api.Client

	logger *slog.Logger
}

// NewOllama creates a new Ollama instance with the specified host URL and model name. The host
// parameter should be a valid URL pointing to an Ollama server. If the provided host URL is invalid,
// the function will panic.
func NewOllama(host, model string, params Parameters, logger *slog.Logger) Ollama {
	u, err := url.Parse(host)
	if err != nil {
		panic(err)
	}

	return Ollama{
		host:   host,
		model:  model,
		params: params,
		client: api.NewClient(u, &http.Client{}),
		logger: logger.With(slog.String("module", "ollama")),
	}
}

// Chat sends a chat message to the Ollama API.
func (o Ollama) Chat(messages []string) (string, error) {
	msgs := make([]api.Message, len(messages))
	for i, msg := range messages {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs[i] = api.Message{
			Role:    role,
			Content: msg,
		}
	}

	req := o.chatRequest(msgs)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var result strings.Builder

	if err := o.client.Chat(ctx, &req, func(res api.ChatResponse) error {
		result.WriteString(res.Message.Content)
		return nil
	}); err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}

	return result.String(), nil
}

func (o Ollama) chatRequest(messages []api.Message) api.ChatRequest {
	req := api.ChatRequest{
		Model:    o.model,
		Messages: messages,
	}

	opts := make(map[string]any)

	stream := false
	req.Stream = &stream

	if o.params.Temperature != nil {
		opts["temperature"] = *o.params.Temperature
	}
	if o.params.Seed != nil {
		opts["seed"] = *o.params.Seed
	}
	if o.params.Stop != nil {
		opts["stop"] = o.params.Stop
	}
	if o.params.TopK != nil {
		opts["top_k"] = *o.params.TopK
	}
	if o.params.TopP != nil {
		opts["top_p"] = *o.params.TopP
	}
	if o.params.MinP != nil {
		opts["min_p"] = *o.params.MinP
	}
	if o.params.IncludeReasoning != nil {
		req.Think = &api.ThinkValue{Value: *o.params.IncludeReasoning}
	}

	req.Options = opts

	return req
}
