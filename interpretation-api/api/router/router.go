package router

import (
	"interpretation-api/api/endpoints"

	"github.com/gin-gonic/gin"
)

func SetupRouter() *gin.Engine {
	r := gin.Default()

	r.GET("/", endpoints.SystemRoot)
	r.GET("/inference-models", endpoints.GetInferenceModels)
	r.GET("/llm-models", endpoints.GetLLMModels)
	r.POST("/interpret", endpoints.Interpret)

	r.StaticFile("/favicon.ico", "./favicon.ico")

	r.StaticFile("/openapi.yaml", "./docs/openapi.yaml")
	r.GET("/docs", endpoints.SwaggerUI)

	return r
}
