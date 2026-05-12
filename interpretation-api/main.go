package main

import (
	"interpretation-api/api/router"
	"interpretation-api/core"
)

func main() {
	core.LoadConfig()

	r := router.SetupRouter()
	r.Run("127.0.0.1:8080")
}
