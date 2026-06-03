import os
import json
import csv
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
raw_results_path = os.path.join(results_dir, "eval_testing_raw.json")
csv_output_path = os.path.join(results_dir, "comparison_matrix.csv")
md_output_path = os.path.join(current_dir, "evaluation_summary.md")

# Clean/map LLM IDs to display names
MODEL_DISPLAY_NAMES = {
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B A4B",
    "qwen/qwen3-vl-32b-instruct": "Qwen3 VL 32B",
    "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3 VL 235B A22B",
    "qwen/qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B A3B"
}

def clean_model_name(name):
    return MODEL_DISPLAY_NAMES.get(name, name.split("/")[-1])

def process_results():
    if not os.path.exists(raw_results_path):
        print(f"[!] Error: Raw G-Eval results not found at {raw_results_path}")
        print("[!] Make sure to run evaluate_testing.py first.")
        return

    print("[*] Reading raw G-Eval results...")
    with open(raw_results_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    evaluator_model = raw_data.get("evaluator_model", "openai/gpt-5")
    runs = raw_data.get("results", [])

    # Group by LLM model and prompt style
    # Combination key: (model, style)
    combinations = {}

    for run in runs:
        model = clean_model_name(run["llm_model"])
        style = run["prompt_style"]
        
        combo_key = (model, style)
        if combo_key not in combinations:
            combinations[combo_key] = {
                "model": model,
                "style": style,
                "metrics": {}
            }

        for metric in run["metrics"]:
            m_name = metric["name"]
            score = metric["score"]
            if m_name not in combinations[combo_key]["metrics"]:
                combinations[combo_key]["metrics"][m_name] = []
            combinations[combo_key]["metrics"][m_name].append(score)

    # Process and average metrics for each combination
    compiled_data = []
    metric_names = set()

    for (model, style), data in combinations.items():
        row = {
            "LLM Model": model,
            "Prompting Style": style.capitalize()
        }
        
        total_scores = []
        for m_name, scores in data["metrics"].items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            row[m_name] = round(avg_score, 4)
            metric_names.add(m_name)
            total_scores.append(avg_score)
            
        row["Overall Average"] = round(sum(total_scores) / len(total_scores) if total_scores else 0.0, 4)
        compiled_data.append(row)

    metric_names = sorted(list(metric_names))

    # Sort compiled_data by Overall Average descending
    compiled_data.sort(key=lambda x: x["Overall Average"], reverse=True)

    # --- 1. Export comparison matrix to CSV ---
    print(f"[*] Exporting CSV comparison matrix to {csv_output_path}...")
    csv_headers = ["LLM Model", "Prompting Style"] + metric_names + ["Overall Average"]
    
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in compiled_data:
            writer.writerow(row)

    # --- 1B. Export simplified flat JSON report of every single evaluated metric ---
    flat_report_path = os.path.join(results_dir, "g_eval_full_report.json")
    print(f"[*] Exporting flat JSON report to {flat_report_path}...")
    flat_report = []
    for run in runs:
        model_name = clean_model_name(run["llm_model"])
        style = run["prompt_style"]
        filename_base = run["filename"]
        for metric in run["metrics"]:
            flat_report.append({
                "Filename": filename_base,
                "LLM Model": model_name,
                "Prompting Style": style.capitalize(),
                "Response Style": run["response_style"],
                "Metric Name": metric["name"],
                "Score": metric["score"],
                "Passed": metric["passed"],
                "Reason": metric["reason"]
            })
    with open(flat_report_path, "w", encoding="utf-8") as f:
        json.dump(flat_report, f, indent=2, ensure_ascii=False)

    # --- 2. Aggregate metrics for High-Level Rankings ---
    # A. Aggregates by LLM
    llm_aggregates = {}
    for row in compiled_data:
        llm = row["LLM Model"]
        if llm not in llm_aggregates:
            llm_aggregates[llm] = []
        llm_aggregates[llm].append(row["Overall Average"])

    llm_rankings = []
    for llm, averages in llm_aggregates.items():
        llm_rankings.append({
            "LLM Model": llm,
            "Average Score": round(sum(averages) / len(averages), 4)
        })
    llm_rankings.sort(key=lambda x: x["Average Score"], reverse=True)

    # B. Aggregates by Prompting Style
    style_aggregates = {}
    for row in compiled_data:
        style = row["Prompting Style"]
        if style not in style_aggregates:
            style_aggregates[style] = []
        style_aggregates[style].append(row["Overall Average"])

    style_rankings = []
    for style, averages in style_aggregates.items():
        style_rankings.append({
            "Prompting Style": style,
            "Average Score": round(sum(averages) / len(averages), 4)
        })
    style_rankings.sort(key=lambda x: x["Average Score"], reverse=True)

    # Best overall configuration
    best_config = compiled_data[0] if compiled_data else None

    # --- 3. Generate Markdown Report ---
    print(f"[*] Generating Markdown Report: {md_output_path}...")
    
    md_content = f"""# Automated G-Eval Comparison Summary Report

This automated report compares the **5 LLM Models** and **2 Prompting Styles** evaluated against the test combinations cached in the pipeline. It is based on evaluations performed by **{evaluator_model}** via OpenRouter.

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Evaluator: {evaluator_model}

---

## 🏆 Recommendation for Psychologist Validation

Based on the automated G-Eval G-5 evaluation metrics, here is the recommended combination for publicity and psychologist assessment:

> [!TIP]
> **Recommended Configuration**: **{best_config['LLM Model']}** using **{best_config['Prompting Style']} Prompting**
> * **Overall G-Eval Score**: **{best_config['Overall Average']:.4f}** / 1.00
> * **Image Coherence**: {best_config.get('Koherensi Gambar', 'N/A')}
> * **Psychological Accuracy**: {best_config.get('Akurasi Penilaian Psikologis', 'N/A')}
> * **Psychologist Persona**: {best_config.get('Persona Psikolog', 'N/A')}
>
> Using this model and prompt style ensures the highest consistency, alignment with traits, and native psychologist tone, allowing you to present **only this best configuration** (exactly 2 pages) to the psychologist rather than all 20 combinations.

---

## 📊 Summary of LLM Model Performance

This table ranks the 5 LLM models based on their average score across all prompt styles and evaluation criteria.

| Rank | LLM Model | Average Score (Out of 1.0) |
|:---:|:---|:---:|
"""
    for rank, item in enumerate(llm_rankings, 1):
        md_content += f"| {rank} | **{item['LLM Model']}** | {item['Average Score']:.4f} |\n"

    md_content += """
---

## 📝 Prompting Style Comparison

This table compares the performance of **Roleplay Prompting** vs **Standard Prompting** across all LLMs.

| Prompting Style | Average Score (Out of 1.0) |
|:---|:---:|
"""
    for item in style_rankings:
        md_content += f"| {item['Prompting Style']} Prompting | {item['Average Score']:.4f} |\n"

    md_content += """
---

## 🏁 Detailed Combination Grid Matrix

This full matrix displays individual scores for every metrics criteria across all 10 configurations. Detailed CSV records are exported to [comparison_matrix.csv](file:///e:/ITS/Tugas%20Akhir/personality-interpretation/eval_testing_g-eval/results/comparison_matrix.csv).

| LLM Model | Prompting Style | Koherensi Gambar | Akurasi Psikologi | Persona Psikolog | Overall Average |
|:---|:---|:---:|:---:|:---:|:---:|
"""
    for row in compiled_data:
        md_content += f"| {row['LLM Model']} | {row['Prompting Style']} | {row.get('Koherensi Gambar', 'N/A'):.4f} | {row.get('Akurasi Penilaian Psikologis', 'N/A'):.4f} | {row.get('Persona Psikolog', 'N/A'):.4f} | **{row['Overall Average']:.4f}** |\n"

    md_content += f"""
---
*Report generated automatically. CSV dataset available in [results/comparison_matrix.csv](file:///e:/ITS/Tugas%20Akhir/personality-interpretation/eval_testing_g-eval/results/comparison_matrix.csv).*
"""

    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[+] Summary processed successfully! Markdown saved to {md_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    process_results()
