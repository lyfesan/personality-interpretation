package services

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	openrouter "github.com/revrost/go-openrouter"

	"interpretation-api/core"
	"interpretation-api/schemas"
)

func IsLLMModelAllowed(modelID string) bool {
	data, err := os.ReadFile("config/llm_models.json")
	if err != nil {
		return false
	}

	var models []struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &models); err != nil {
		return false
	}

	for _, m := range models {
		if m.ID == modelID {
			return true
		}
	}
	return false
}

func GetResponseStyles() ([]schemas.ResponseStyle, error) {
	data, err := os.ReadFile("config/response_styles.json")
	if err != nil {
		return nil, err
	}

	var styles []schemas.ResponseStyle
	if err := json.Unmarshal(data, &styles); err != nil {
		return nil, err
	}
	return styles, nil
}

func GenerateInterpretation(llmModel string, imageBase64 string, traits schemas.OCEANTraits, styleID string) (string, error) {
	if styleID == "" {
		styleID = "comprehensive_id"
	}

	styles, err := GetResponseStyles()
	if err != nil {
		return "", fmt.Errorf("failed to load response styles: %w", err)
	}

	templateFile := ""
	for _, s := range styles {
		if s.ID == styleID {
			templateFile = s.TemplateFile
			break
		}
	}

	// Fallback to comprehensive_id if not found
	if templateFile == "" {
		templateFile = "prompts/comprehensive_id.txt"
	}

	promptTemplateBytes, err := os.ReadFile(templateFile)
	if err != nil {
		return "", fmt.Errorf("failed to read prompt template %s: %w", templateFile, err)
	}

	prompt := fmt.Sprintf(string(promptTemplateBytes), traits.Openness, traits.Conscientiousness, traits.Extraversion, traits.Agreeableness, traits.Neuroticism)

	// Ensure the base64 string has the correct prefix for OpenRouter
	imageUrl := imageBase64
	if !strings.HasPrefix(imageUrl, "data:image") {
		imageUrl = "data:image/jpeg;base64," + imageUrl
	}

	client := openrouter.NewClient(
		core.AppConfig.OpenRouterKey,
		openrouter.WithXTitle("Big Five Personality Interpreter"),
		openrouter.WithHTTPReferer(core.AppConfig.AppUrl),
	)

	req := openrouter.ChatCompletionRequest{
		Model:     llmModel,
		MaxTokens: 1000,
		Messages: []openrouter.ChatCompletionMessage{
			openrouter.UserMessageWithImage(prompt, imageUrl),
		},
	}

	resp, err := client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return "", fmt.Errorf("OpenRouter API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from OpenRouter")
	}

	return resp.Choices[0].Message.Content.Text, nil
}
