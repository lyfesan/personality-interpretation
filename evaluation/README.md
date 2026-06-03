# DeepEval Evaluation Pipeline

This directory contains the custom evaluation pipeline using **DeepEval** to score the multimodal outputs of the Personality Interpretation pipeline. 

Because standard LLM-as-a-judge metrics (like *Faithfulness*) don't perfectly map to Multimodal Image inputs, we implemented a custom Dynamic Multimodal G-Eval metric.

## Configuration

1. Make sure your `.env` is loaded with your `OPENROUTER_API_KEY`. The custom evaluator sends the Image and Traits directly to GPT-4o (or another model) via OpenRouter.
2. Edit `metrics_config.json` to tweak the evaluation criteria and pass thresholds. You can add new metrics here dynamically without changing the code!

## Running the Evaluation

1. Use your Gradio app to interpret personality traits.
2. Click the **Export Result as JSON** button.
3. Move the downloaded `.json` files into the `evaluation/test_data/` folder.
4. Run the evaluation script:
   ```bash
   python evaluate_pipeline.py
   ```

DeepEval will output a detailed breakdown of scores and reasoning for each interpretation and image pair based on the dynamic criteria.
