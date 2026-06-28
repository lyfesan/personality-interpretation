import os
import sys
import json
from glob import glob
from datetime import datetime
from dotenv import load_dotenv

# Set up paths to load evaluation modules dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
evaluation_dir = os.path.join(parent_dir, "evaluation")
json_dir = os.path.join(current_dir, "json")

sys.path.append(evaluation_dir)

# Load environment variables from the evaluation folder's .env file
dotenv_path = os.path.join(evaluation_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[*] Loaded .env from: {dotenv_path}")
else:
    print(f"[!] Warning: .env file not found at {dotenv_path}. Make sure OPENROUTER_API_KEY is in your environment.")

# Check for OpenRouter API Key
if not os.environ.get("OPENROUTER_API_KEY"):
    print("[!] Critical Error: OPENROUTER_API_KEY is not set.")
    sys.exit(1)

from deepeval.test_case import LLMTestCase
from custom_metrics import DynamicMultimodalMetric

def load_metrics_config():
    config_path = os.path.join(evaluation_dir, "metrics_config.json")
    if not os.path.exists(config_path):
        print(f"[!] Error: metrics_config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    print("=" * 60)
    print("         G-Eval LLM Automated Comparison Pipeline")
    print("=" * 60)

    # 1. Load configuration and metrics
    config = load_metrics_config()
    evaluator_model = config.get("evaluator_model", "openai/gpt-5")
    print(f"[*] Evaluator Model: {evaluator_model}")
    current_metric_names = [m["name"] for m in config.get("metrics", [])]
    print(f"[*] Metrics to run: {current_metric_names}")

    # 2. Get list of generated JSON combinations from testing/json
    json_files = glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        print(f"[!] Error: No cached combinations found in {json_dir}")
        print("[!] Please run generate_samples.py first to cache all combinations.")
        sys.exit(1)

    print(f"[*] Found {len(json_files)} cached combinations to evaluate.")
    print("-" * 60)

    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    raw_results_path = os.path.join(results_dir, "eval_testing_raw.json")

    # Load existing checkpoint/results to resume
    cached_results = {}
    if os.path.exists(raw_results_path):
        try:
            with open(raw_results_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                cached_runs = checkpoint_data.get("results", [])
                for run in cached_runs:
                    cached_results[run["filename"]] = run
            print(f"[*] Loaded G-Eval checkpoint containing {len(cached_results)} previously evaluated runs.")
        except Exception as e:
            print(f"[!] Warning: Failed to load G-Eval checkpoint: {e}")

    results = []

    # 3. Process and evaluate each combination
    for idx, file_path in enumerate(json_files, 1):
        filename = os.path.basename(file_path)
        print(f"[{idx}/{len(json_files)}] Evaluating: {filename}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("parameters", {})
        inference_model = params.get("inference_model", "vit-b16-augreg-in21k")
        llm_model = params.get("llm_model", "unknown")
        response_style = params.get("response_style", "unknown")

        # Determine prompt style classification
        if "roleplay" in response_style.lower() or "roleplay" in filename.lower():
            prompt_style = "roleplay"
        else:
            prompt_style = "standard"

        # Check if we can load this from checkpoint
        use_checkpoint = False
        if filename in cached_results:
            cached_case = cached_results[filename]
            cached_metrics = {m["name"]: m for m in cached_case.get("metrics", [])}
            
            # Verify all metrics exist and none failed (payment/client errors)
            is_valid = True
            for name in current_metric_names:
                if name not in cached_metrics:
                    is_valid = False
                    break
                reason = cached_metrics[name].get("reason", "")
                if "failed" in reason.lower() or "client error" in reason.lower() or "payment" in reason.lower():
                    is_valid = False
                    break
            
            if is_valid:
                use_checkpoint = True
                case_scores = cached_case
                print("    [-] Loaded from G-Eval checkpoint.")

        if not use_checkpoint:
            res_block = data.get("results", {})
            traits = res_block.get("predictions", {})
            interpretation = res_block.get("interpretation", "")
            image_b64 = data.get("image_base64", "")

            # Prepare DeepEval test case
            input_data = {
                "traits": traits,
                "image_base64": image_b64
            }
            test_case = LLMTestCase(
                input=json.dumps(input_data),
                actual_output=interpretation
            )

            case_scores = {
                "filename": filename,
                "llm_model": llm_model,
                "response_style": response_style,
                "prompt_style": prompt_style,
                "inference_model": inference_model,
                "metrics": []
            }

            # Initialize and evaluate each metric dynamically
            for m in config.get("metrics", []):
                metric = DynamicMultimodalMetric(
                    name=m["name"],
                    criteria=m["criteria"],
                    threshold=m.get("threshold", 0.5),
                    model=evaluator_model,
                    use_image=m.get("use_image", False)
                )

                print(f"    - Running Metric: {m['name']}...", end="", flush=True)
                try:
                    score = metric.measure(test_case)
                    reason = metric.reason
                    passed = metric.is_successful()
                    print(f" Completed. Score: {score:.2f} | Passed: {passed}")
                except Exception as e:
                    score = 0.0
                    reason = f"Execution failed: {str(e)}"
                    passed = False
                    print(f" Failed! Error: {e}")

                case_scores["metrics"].append({
                    "name": m["name"],
                    "score": score,
                    "threshold": m.get("threshold", 0.5),
                    "passed": passed,
                    "reason": reason
                })

        results.append(case_scores)

        # Save checkpoint to disk immediately after evaluating the combination
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "evaluator_model": evaluator_model,
            "total_evaluated": len(results),
            "results": results
        }
        with open(raw_results_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print("-" * 60)

    print(f"[+] Evaluation finished! Raw results successfully saved to: {raw_results_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
