package main

import (
	"interpretation-api/api/router"
	"interpretation-api/core"
	"interpretation-api/services"
	"os"
)

func main() {
	core.LoadConfig()

	// Start daily keep-alive check for Hugging Face inference API
	services.StartDailyKeepAlive(core.AppConfig.InferenceApiUrl)

	r := router.SetupRouter()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	r.Run(":" + port)
}
