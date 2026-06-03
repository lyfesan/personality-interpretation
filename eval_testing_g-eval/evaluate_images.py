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
sys.path.append(evaluation_dir)

# Ensure test_data directory exists inside the new folder
test_data_dir = os.path.join(current_dir, "test_data")
os.makedirs(test_data_dir, exist_ok=True)

# Load environment variables from the evaluation folder's .env file
dotenv_path = os.path.join(evaluation_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

from deepeval.test_case import LLMTestCase
from custom_metrics import DynamicMultimodalMetric

def load_metrics_config():
    config_path = os.path.join(evaluation_dir, "metrics_config.json")
    if not os.path.exists(config_path):
        print(f"[!] Error: metrics_config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_custom_evaluation():
    print("=" * 60)
    print("        G-Eval Custom Sample Image Evaluation")
    print("=" * 60)

    # 1. Check for files in test_data
    export_files = glob(os.path.join(test_data_dir, "*.json"))
    if not export_files:
        print(f"[!] No custom sample JSON files found in '{test_data_dir}/'")
        print("[*] Please export your custom interpretations from Gradio and place them here.")
        print("    File pattern should be: *.json")
        print("=" * 60)
        return

    # 2. Check for OpenRouter API Key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[!] Critical Error: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    # 3. Load configurations
    config = load_metrics_config()
    evaluator_model = config.get("evaluator_model", "openai/gpt-5")
    print(f"[*] Evaluator Model: {evaluator_model}")
    print(f"[*] Found {len(export_files)} custom test case(s) to evaluate.")
    print("-" * 60)

    results = []

    # 4. Evaluate each custom test case
    for idx, file_path in enumerate(export_files, 1):
        filename = os.path.basename(file_path)
        print(f"[{idx}/{len(export_files)}] Evaluating Custom Sample: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("parameters", {})
        llm_model = params.get("llm_model", "unknown")
        response_style = params.get("response_style", "unknown")

        res_block = data.get("results", {})
        traits = res_block.get("predictions", {})
        interpretation = res_block.get("interpretation", "")
        image_b64 = data.get("image_base64", "")

        # Prepare test case
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
            "metrics": []
        }

        for m in config.get("metrics", []):
            metric = DynamicMultimodalMetric(
                name=m["name"],
                criteria=m["criteria"],
                threshold=m.get("threshold", 0.5),
                model=evaluator_model,
                use_image=m.get("use_image", False)
            )

            print(f"    - Evaluating: {m['name']}...", end="", flush=True)
            try:
                score = metric.measure(test_case)
                reason = metric.reason
                passed = metric.is_successful()
                print(f" Score: {score:.2f} | Passed: {passed}")
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
        print("-" * 60)

    # 5. Output Report
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"custom_image_eval_{timestamp}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluator_model": evaluator_model,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("                  CUSTOM EVALUATION SUMMARY")
    print("=" * 60)
    for case in results:
        print(f"\nFile: {case['filename']}")
        print(f"Model: {case['llm_model']} | Style: {case['response_style']}")
        print("-" * 40)
        for m in case["metrics"]:
            status = "PASSED" if m["passed"] else "FAILED"
            print(f"  * {m['name']}: {m['score']:.2f} / 1.00 [{status}]")
            print(f"    Reason: {m['reason']}")

    print("\n" + "=" * 60)
    print(f"[+] Full custom report saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_custom_evaluation()
