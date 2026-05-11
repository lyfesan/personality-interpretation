package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"api_name":    "Big Five Personality Interpretation API",
			"description": "API for generating personality interpretation based on apparent Big Five traits from image",
			"version":     "0.1.0",
			"status":      "online",
		})
	})

	r.Run("127.0.0.1:8080")
}
