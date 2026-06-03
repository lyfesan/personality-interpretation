package main

import (
	"interpretation-api/api/router"
	"interpretation-api/core"
	"os"
)

func main() {
	core.LoadConfig()

	r := router.SetupRouter()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	r.Run(":" + port)
}
