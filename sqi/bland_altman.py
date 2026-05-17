import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Compute Bland-Altman
# -----------------------------
def compute_bland_altman(ppg, ecg):
    ppg_values = ppg.iloc[:, 1].to_numpy()
    ecg_values = ecg.iloc[:, 1].to_numpy()

    if len(ppg_values) != len(ecg_values):
        raise ValueError("PPG and ECG must have the same length.")

    mean_vals = (ppg_values + ecg_values) / 2
    diff = ppg_values - ecg_values

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    results = pd.DataFrame({
        "mean_ppg_ecg": mean_vals,
        "difference_ppg_minus_ecg": diff,
        "bias": bias,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower
    })

    return results


# -----------------------------
# Plot
# -----------------------------
def plot_bland_altman(results):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        results["mean_ppg_ecg"],
        results["difference_ppg_minus_ecg"],
        alpha=0.7
    )

    bias = results["bias"].iloc[0]
    loa_upper = results["loa_upper"].iloc[0]
    loa_lower = results["loa_lower"].iloc[0]

    plt.axhline(bias, linestyle="--", label=f"Bias = {bias:.3f}")
    plt.axhline(loa_upper, linestyle="--", label=f"Upper LoA = {loa_upper:.3f}")
    plt.axhline(loa_lower, linestyle="--", label=f"Lower LoA = {loa_lower:.3f}")

    plt.xlabel("Mean of PPG and ECG")
    plt.ylabel("Difference (PPG - ECG)")
    plt.title("Bland-Altman Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# Run (standalone CLI behaviour preserved when invoked directly)
# -----------------------------
if __name__ == "__main__":
    ppg = pd.read_csv("sample_ppg.csv")
    ecg = pd.read_csv("sample_ecg.csv")

    results = compute_bland_altman(ppg, ecg)

    print(results)

    results.to_csv("bland_altman_results.csv", index=False)

    plot_bland_altman(results)
