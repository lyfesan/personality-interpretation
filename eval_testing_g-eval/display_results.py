import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
csv_path = os.path.join(results_dir, "comparison_matrix.csv")
chart_path = os.path.join(results_dir, "comparison_chart.png")

def display_and_visualize():
    if not os.path.exists(csv_path):
        print(f"[!] Error: Comparison matrix CSV not found at {csv_path}")
        print("[!] Please run generate_summary.py first.")
        return

    # 1. Load CSV using pandas
    try:
        import pandas as pd
    except ImportError:
        print("[!] Warning: 'pandas' library is not installed. Installing it or using standard parsing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    df = pd.read_csv(csv_path)

    print("=" * 75)
    print("                G-Eval LLM Comparison Matrix (Pandas Output)")
    print("=" * 75)
    # Output formatted string
    print(df.to_string(index=False))
    print("=" * 75)

    # 2. Check and generate Matplotlib chart
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[*] Matplotlib not found. Attempting to install 'matplotlib' via pip for visualization...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[!] Warning: Could not install matplotlib: {e}. Skipping chart generation.")
            return

    try:
        print("[*] Generating comparison chart using Matplotlib...")
        # Prepare data for grouped bar chart
        # We need columns: LLM Model, Prompting Style, Overall Average
        # We pivot the table to have LLMs as index, Styles as columns, and Overall Average as values
        pivot_df = df.pivot(index="LLM Model", columns="Prompting Style", values="Overall Average")

        # Set up plot styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot grouped bar chart
        pivot_df.plot(kind="bar", ax=ax, width=0.6, color=["#3B82F6", "#10B981"])

        ax.set_title("G-Eval Overall Score Comparison (GPT-5 Evaluator)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("LLM Model", fontsize=12, labelpad=10)
        ax.set_ylabel("Overall Score (0.0 - 1.0)", fontsize=12, labelpad=10)
        ax.set_ylim(0, 1.05)
        
        # Add labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", label_type="edge", padding=3, fontsize=9)

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        # Save plot
        plt.savefig(chart_path, dpi=300)
        print(f"[+] Comparison chart successfully saved to: {chart_path}")
        print("=" * 75)
    except Exception as e:
        print(f"[!] Error generating chart: {e}")

if __name__ == "__main__":
    display_and_visualize()
