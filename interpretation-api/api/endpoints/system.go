package endpoints

import (
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

func SystemRoot(c *gin.Context) {
	data, err := os.ReadFile("config/metadata.json")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to read API metadata",
		})
		return
	}

	c.Data(http.StatusOK, "application/json", data)
}

func GetLLMModels(c *gin.Context) {
	data, err := os.ReadFile("config/llm_models.json")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to read LLM models configuration",
		})
		return
	}

	c.Data(http.StatusOK, "application/json", data)
}
