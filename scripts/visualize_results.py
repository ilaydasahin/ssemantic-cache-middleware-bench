import json
import os
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

# Set publication style
plt.style.use("default")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300


def generate_pgfplots(df_agg, output_dir):
    """Generate PGFPlots code for easy inclusion in LaTeX."""
    models = df_agg["embeddingModel"].unique()

    with open(os.path.join(output_dir, "pgfplots_code.txt"), "w") as f:
        f.write("% --- HIT RATE VS THRESHOLD ---\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("    xlabel={Similarity Threshold $\\theta$},\n")
        f.write("    ylabel={Cache Hit Rate (\\%)},\n")
        f.write("    legend pos=south west,\n")
        f.write("    grid=major,\n")
        f.write("    width=0.8\\linewidth, height=0.6\\linewidth\n")
        f.write("]\n")

        for model in models:
            model_df = df_agg[df_agg["embeddingModel"] == model].sort_values(
                "threshold"
            )
            f.write(f"\\addplot coordinates {{\n")
            for _, row in model_df.iterrows():
                f.write(f"    ({row['threshold']:.2f}, {row['hitRate_mean']:.2f})\n")
            f.write("};\n")
            f.write(f"\\addlegendentry{{{model}}}\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n\n")

        f.write("% --- ROUGE-L VS THRESHOLD ---\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("    xlabel={Similarity Threshold $\\theta$},\n")
        f.write("    ylabel={ROUGE-L Score},\n")
        f.write("    legend pos=north east,\n")
        f.write("    grid=major,\n")
        f.write("    width=0.8\\linewidth, height=0.6\\linewidth\n")
        f.write("]\n")

        for model in models:
            model_df = df_agg[df_agg["embeddingModel"] == model].sort_values(
                "threshold"
            )
            f.write(f"\\addplot coordinates {{\n")
            for _, row in model_df.iterrows():
                f.write(f"    ({row['threshold']:.2f}, {row['avgRougeL_mean']:.4f})\n")
            f.write("};\n")
            f.write(f"\\addlegendentry{{{model}}}\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")


def plot_metrics(df_agg, output_dir):
    # 1. Hit Rate vs Threshold
    plt.figure(figsize=(8, 6))
    for model in df_agg["embeddingModel"].unique():
        model_df = df_agg[df_agg["embeddingModel"] == model]
        plt.errorbar(
            model_df["threshold"],
            model_df["hitRate_mean"],
            yerr=model_df["hitRate_std"],
            marker="o",
            label=model,
            capsize=5,
        )

    plt.xlabel(r"Similarity Threshold $\theta$")
    plt.ylabel("Cache Hit Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hit_rate_vs_threshold.pdf"))
    plt.close()

    # 2. ROUGE-L vs Threshold
    plt.figure(figsize=(8, 6))
    for model in df_agg["embeddingModel"].unique():
        model_df = df_agg[df_agg["embeddingModel"] == model]
        plt.errorbar(
            model_df["threshold"],
            model_df["avgRougeL_mean"],
            yerr=model_df["avgRougeL_std"],
            marker="s",
            label=model,
            capsize=5,
        )

    plt.xlabel(r"Similarity Threshold $\theta$")
    plt.ylabel("Semantic Fidelity (ROUGE-L)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rougel_vs_threshold.pdf"))
    plt.close()

    # 3. Latency (p99) vs Threshold
    plt.figure(figsize=(8, 6))
    for model in df_agg["embeddingModel"].unique():
        model_df = df_agg[df_agg["embeddingModel"] == model]
        plt.errorbar(
            model_df["threshold"],
            model_df["p99LatencyMs_mean"],
            yerr=model_df["p99LatencyMs_std"],
            marker="^",
            label=model,
            capsize=5,
        )

    plt.xlabel(r"Similarity Threshold $\theta$")
    plt.ylabel("p99 Hit Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p99latency_vs_threshold.pdf"))
    plt.close()


def plot_heatmap(df, output_dir):
    """Generate a heatmap of Hit Rate vs. Threshold and Model."""
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot_table(
        index="embeddingModel", columns="threshold", values="hitRate_mean"
    )
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Heatmap: Cache Hit Rate (%) by Model and Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_hitrate.pdf"))
    plt.close()


def plot_cumulative_hit_rate(results_dir, output_dir):
    """Analyze .logs.jsonl files to show cache 'warming' over time."""
    plt.figure(figsize=(10, 6))
    
    log_files = [f for f in os.listdir(results_dir) if f.endswith(".logs.jsonl")]
    if not log_files:
        return

    # Just take a representative sample or aggregate if many
    for log_file in log_files[:3]:  # Visualize first 3 for clarity
        hits = []
        with open(os.path.join(results_dir, log_file), "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    hits.append(1 if data.get("isHit") else 0)
                except:
                    continue
        
        if not hits:
            continue
            
        cumulative_hits = np.cumsum(hits)
        hit_rate_over_time = [
            (h / (i + 1)) * 100 for i, h in enumerate(cumulative_hits)
        ]
        plt.plot(hit_rate_over_time, label=log_file.replace(".logs.jsonl", ""), alpha=0.7)

    plt.xlabel("Query Sequence Number")
    plt.ylabel("Cumulative Hit Rate (%)")
    plt.title("Cache Warming Profile: Cumulative Hit Rate over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cache_warming_profile.pdf"))
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_results.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    all_results_path = os.path.join(results_dir, "all_results.json")

    if not os.path.exists(all_results_path):
        print(f"File not found: {all_results_path}")
        sys.exit(1)

    with open(all_results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Aggregate data
    metrics = ["hitRate", "avgRougeL", "p99LatencyMs", "costSavingsPercent"]
    df_agg = (
        df.groupby(["embeddingModel", "threshold"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_agg.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else col[0]
        for col in df_agg.columns.values
    ]

    plot_metrics(df_agg, results_dir)
    plot_heatmap(df_agg, results_dir)
    plot_cumulative_hit_rate(results_dir, results_dir)
    generate_pgfplots(df_agg, results_dir)

    print(f"Visualizations and PGFPlots code saved to {results_dir}")


if __name__ == "__main__":
    main()
