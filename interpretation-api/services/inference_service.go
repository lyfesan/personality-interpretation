package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"interpretation-api/core"
	"interpretation-api/schemas"
)

func GetInferenceModels() ([]byte, error) {
	url := fmt.Sprintf("%s/models", core.AppConfig.InferenceApiUrl)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference API returned status %d", resp.StatusCode)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return bodyBytes, nil
}

func PredictBigFive(modelType string, imageBase64 string) (*schemas.InferenceResponse, error) {
	url := fmt.Sprintf("%s/predict_base64", core.AppConfig.InferenceApiUrl)

	reqBody := schemas.InferenceRequest{
		ModelType:   modelType,
		ImageBase64: imageBase64,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("inference API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var inferenceResp schemas.InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&inferenceResp); err != nil {
		return nil, err
	}

	return &inferenceResp, nil
}
