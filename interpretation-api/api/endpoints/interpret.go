package endpoints

import (
	"encoding/base64"
	"io"
	"net/http"
	"strings"

	"interpretation-api/schemas"
	"interpretation-api/services"

	"github.com/gin-gonic/gin"
)

func GetInferenceModels(c *gin.Context) {
	result, err := services.GetInferenceModels()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.Data(http.StatusOK, "application/json", result)
}

func Interpret(c *gin.Context) {
	var req schemas.InterpretRequest

	if strings.HasPrefix(c.ContentType(), "multipart/form-data") {
		req.InferenceModel = c.PostForm("inference_model")
		req.LLMModel = c.PostForm("llm_model")

		file, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Missing 'image' file in form data"})
			return
		}

		fileContent, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open image file"})
			return
		}
		defer fileContent.Close()

		bytes, err := io.ReadAll(fileContent)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read image file"})
			return
		}

		req.Image = base64.StdEncoding.EncodeToString(bytes)

	} else {
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Invalid request payload: " + err.Error(),
			})
			return
		}
	}

	if req.InferenceModel == "" || req.LLMModel == "" || req.Image == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "inference_model, llm_model, and image are required"})
		return
	}

	// 0. Validate if the LLM model is allowed
	if !services.IsLLMModelAllowed(req.LLMModel) {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "LLM model '" + req.LLMModel + "' is not allowed or not found in configuration",
		})
		return
	}

	// 1. Send image to inference-api to get Big Five traits
	inferenceResp, err := services.PredictBigFive(req.InferenceModel, req.Image)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to get predictions from inference API: " + err.Error(),
		})
		return
	}

	// 2. Use traits and image to get interpretation from LLM
	interpretation, err := services.GenerateInterpretation(req.LLMModel, req.Image, inferenceResp.Predictions)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to generate interpretation from LLM: " + err.Error(),
		})
		return
	}

	// 3. Return final result
	c.JSON(http.StatusOK, gin.H{
		"model_used":     inferenceResp.ModelUsed,
		"predictions":    inferenceResp.Predictions,
		"llm_used":       req.LLMModel,
		"interpretation": interpretation,
	})
}
