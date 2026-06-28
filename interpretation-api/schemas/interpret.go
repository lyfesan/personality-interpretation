package schemas

type ResponseStyle struct {
	ID                 string `json:"id"`
	Name               string `json:"name"`
	Description        string `json:"description"`
	TemplateFile       string `json:"template_file"`
	SystemTemplateFile string `json:"system_template_file,omitempty"`
	Language           string `json:"language"`
}

type InterpretRequest struct {
	InferenceModel string `json:"inference_model" binding:"required"`
	LLMModel       string `json:"llm_model" binding:"required"`
	Image          string `json:"image" binding:"required"`
	StyleID        string `json:"style_id"`
}

type InferenceRequest struct {
	ModelType   string `json:"model_type"`
	ImageBase64 string `json:"image_base64"`
}

type OCEANTraits struct {
	Openness          float64 `json:"Openness"`
	Conscientiousness float64 `json:"Conscientiousness"`
	Extraversion      float64 `json:"Extraversion"`
	Agreeableness     float64 `json:"Agreeableness"`
	Neuroticism       float64 `json:"Neuroticism"`
}

type InferenceResponse struct {
	ModelUsed         string      `json:"model_used"`
	Predictions       OCEANTraits `json:"predictions"`
	CroppedFaceBase64 string      `json:"cropped_face_base64"`
}
