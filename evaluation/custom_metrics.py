import json
import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from openrouter_client import evaluate_with_openrouter

class DynamicMultimodalMetric(BaseMetric):
    """
    A custom DeepEval metric that dynamically evaluates a criteria using OpenRouter.
    It supports Multimodal inputs (parsing the image_base64 from the test case input).
    Metrics can optionally include the image via the 'use_image' config flag.
    """
    def __init__(self, name: str, criteria: str, threshold: float = 0.5,
                 model: str = "openai/gpt-4o", use_image: bool = False, **kwargs):
        self.name = name
        self.evaluation_name = name
        self.criteria = criteria
        self.threshold = threshold
        self.model = model
        self.use_image = use_image
        self.reason = None
        self.score = None
        self.success = False
        self.error = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            input_data = json.loads(test_case.input)
            traits = input_data.get("traits", {})
            image_b64 = input_data.get("image_base64", None)
        except Exception:
            traits = test_case.input
            image_b64 = None

        # Only pass the image to the evaluator if this metric requires it
        send_image = image_b64 if self.use_image else None

        prompt = f"""Anda adalah evaluator NLP ahli yang melakukan penilaian akademis terhadap teks yang dihasilkan oleh sistem.
Ini merupakan bagian dari proyek penelitian universitas tentang interpretasi kepribadian otomatis.
Tugas Anda adalah menilai kualitas teks yang dihasilkan berdasarkan kriteria tertentu.

KRITERIA EVALUASI:
{self.criteria}

DATA INPUT (Skor Trait Kepribadian Big Five, skala 0-1):
{json.dumps(traits, indent=2)}

TEKS YANG DIEVALUASI:
---
{test_case.actual_output}
---

Instruksi:
- Berikan skor pada teks antara 0.0 (sama sekali tidak memenuhi kriteria) dan 1.0 (memenuhi kriteria dengan sempurna).
- Anda HARUS merespons dengan HANYA objek JSON yang valid dengan tepat dua field:
  "score": angka desimal antara 0.0 dan 1.0
  "reason": penjelasan singkat untuk skor yang diberikan (dalam Bahasa Indonesia)

Contoh respons:
{{"score": 0.85, "reason": "Teks secara akurat mencerminkan skor trait dengan sedikit kelalaian."}}
"""
        try:
            response_text = evaluate_with_openrouter(prompt, image_base64=send_image, model=self.model)

            # Extract JSON block from response
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                self.score = float(res.get("score", 0.0))
                self.reason = res.get("reason", "No reason provided")
            else:
                self.score = 0.0
                self.reason = f"Failed to parse JSON from Evaluator LLM. Raw Output: {response_text}"
        except Exception as e:
            self.score = 0.0
            self.reason = f"Evaluation request failed: {str(e)}"

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return self.evaluation_name
