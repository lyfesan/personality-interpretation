package core

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	InferenceApiUrl string
	OpenRouterKey   string
	AppUrl          string
}

var AppConfig *Config

func LoadConfig() {
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found or error loading it. Relying on system environment variables.")
	}

	AppConfig = &Config{
		InferenceApiUrl: os.Getenv("INFERENCE_API_URL"),
		OpenRouterKey:   os.Getenv("OPENROUTER_API_KEY"),
		AppUrl:          os.Getenv("APP_URL"),
	}

	if AppConfig.AppUrl == "" {
		AppConfig.AppUrl = "http://localhost:8080"
	}

	if AppConfig.InferenceApiUrl == "" {
		log.Fatal("INFERENCE_API_URL is required")
	}
}
