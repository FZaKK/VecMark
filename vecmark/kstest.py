import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os

def main():
    """Perform KS test and distribution analysis on two score files"""
    parser = argparse.ArgumentParser(description="KS test and distribution analysis for score files")
    parser.add_argument("--file1", type=str, required=True, help="Path to first CSV file with scores")
    parser.add_argument("--file2", type=str, required=True, help="Path to second CSV file with scores")
    parser.add_argument("--output", type=str, default="./distribution_plot.png", help="Output path for the plot image")
    args = parser.parse_args()

    # Read data
    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)

    scores1 = df1["score"].values
    scores2 = df2["score"].values 

    # ========== Distribution Fitting ==========
    # Fit normal distribution
    param1 = stats.norm.fit(scores1)  # (loc, scale)
    param2 = stats.norm.fit(scores2)

    print("File1 norm params (mean, std):", param1)
    print("File2 norm params (mean, std):", param2)

    # ========== Create Histogram + Fitted Curves ==========
    x = np.linspace(min(scores1.min(), scores2.min()),
                    max(scores1.max(), scores2.max()), 200)

    pdf1 = stats.norm.pdf(x, *param1)
    pdf2 = stats.norm.pdf(x, *param2)

    plt.figure(figsize=(10,6))
    plt.hist(scores1, bins=30, density=True, alpha=0.5, label="File1 scores")
    plt.hist(scores2, bins=30, density=True, alpha=0.5, label="File2 scores")

    plt.plot(x, pdf1, "r-", lw=2, label="File1 fitted norm")
    plt.plot(x, pdf2, "b-", lw=2, label="File2 fitted norm")

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Score Distribution & Normal Fit")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save plot to file instead of displaying
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")
    plt.close()

    # ========== KS Test ==========
    # Two-sample KS test
    ks_stat, p_value = stats.ks_2samp(scores1, scores2)

    print(f"KS Statistic = {ks_stat:.4f}, p-value = {p_value:.4e}")

    # ========== Normality Tests ==========
    # Optional: Normality test for each sample (Shapiro-Wilk test)
    shapiro1 = stats.shapiro(scores1)
    shapiro2 = stats.shapiro(scores2)

    print(f"File1 Shapiro-Wilk test: statistic={shapiro1.statistic:.4f}, p-value={shapiro1.pvalue:.4e}")
    print(f"File2 Shapiro-Wilk test: statistic={shapiro2.statistic:.4f}, p-value={shapiro2.pvalue:.4e}")

if __name__ == "__main__":
    main()