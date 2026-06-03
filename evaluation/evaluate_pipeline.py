import json
import os
import sys
from glob import glob

# Force UTF-8 encoding for rich console output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from custom_metrics import DynamicMultimodalMetric

def load_config():
    with open("metrics_config.json", "r") as f:
        return json.load(f)

def run_evaluation():
    config = load_config()
    evaluator_model = config.get("evaluator_model", "openai/gpt-4o")
    
    # 1. Load exported JSON test cases
    test_cases_dir = "test_data"
    os.makedirs(test_cases_dir, exist_ok=True)
    
    export_files = glob(f"{test_cases_dir}/*.json")
    if not export_files:
        print(f"Warning: No test cases found in '{os.path.abspath(test_cases_dir)}/'.")
        print("Please place the exported JSON files from the Gradio frontend into this directory.")
        return

    print(f"Loaded {len(export_files)} test cases. Initializing DeepEval...\n")
    test_cases = []
    
    for file_path in export_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract data from the frontend's exported JSON schema
        results = data.get("results", {})
        traits = results.get("predictions", {})
        actual_output = results.get("interpretation", "")
        image_b64 = data.get("image_base64", "")
        
        # Package traits and image into a single JSON string for DeepEval's input field
        input_data = {
            "traits": traits,
            "image_base64": image_b64
        }
        
        test_case = LLMTestCase(
            input=json.dumps(input_data),
            actual_output=actual_output
        )
        test_cases.append(test_case)

    # 2. Setup dynamic custom metrics
    metrics = []
    for m in config.get("metrics", []):
        metric = DynamicMultimodalMetric(
            name=m["name"],
            criteria=m["criteria"],
            threshold=m.get("threshold", 0.5),
            model=evaluator_model,
            use_image=m.get("use_image", False)
        )
        metrics.append(metric)

    # 3. Run evaluation
    print(f"Running evaluation using evaluator model: {evaluator_model}...")
    eval_results = evaluate(test_cases, metrics)

    # 4. Save results to JSON file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"eval_{timestamp}.json")

    output = {
        "timestamp": datetime.now().isoformat(),
        "evaluator_model": evaluator_model,
        "num_test_cases": len(test_cases),
        "test_results": []
    }

    # Extract results from DeepEval's returned EvaluationResult object
    for test_result in eval_results.test_results:
        case_result = {
            "test_case": test_result.index,
            "success": test_result.success,
            "input_preview": (test_result.input[:200] + "...") if test_result.input and len(test_result.input) > 200 else test_result.input,
            "actual_output_preview": (test_result.actual_output[:300] + "...") if test_result.actual_output and len(test_result.actual_output) > 300 else test_result.actual_output,
            "metrics": []
        }
        if test_result.metrics_data:
            for md in test_result.metrics_data:
                case_result["metrics"].append({
                    "name": md.name,
                    "score": md.score,
                    "threshold": md.threshold,
                    "passed": md.success,
                    "reason": md.reason,
                    "error": md.error
                })
        output["test_results"].append(case_result)

    # Compute aggregate summary
    aggregate = {}
    for test_result in eval_results.test_results:
        if test_result.metrics_data:
            for md in test_result.metrics_data:
                if md.name not in aggregate:
                    aggregate[md.name] = {"scores": [], "threshold": md.threshold, "passed_count": 0}
                if md.score is not None:
                    aggregate[md.name]["scores"].append(md.score)
                if md.success:
                    aggregate[md.name]["passed_count"] += 1

    output["aggregate"] = {}
    for name, data in aggregate.items():
        total = len(data["scores"])
        output["aggregate"][name] = {
            "average_score": round(sum(data["scores"]) / total, 4) if total > 0 else None,
            "threshold": data["threshold"],
            "pass_rate": f"{data['passed_count']}/{total}",
            "total": total
        }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {os.path.abspath(result_file)}")

if __name__ == "__main__":
    run_evaluation()
