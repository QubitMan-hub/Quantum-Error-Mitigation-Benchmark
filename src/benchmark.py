import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.circuits import trotter_circuit, magnetization, exact_magnetization
from src.noise_model import build_noise_model
from src.pec import pec_expectation
from src.zne import noisy_expectation, zne_expectation

T_EVOLUTION = 1.5
STEP_COUNTS = [1, 2, 4, 8, 12, 16, 24, 32]
SHOTS = 8192
ONE_QUBIT_ERROR = 0.001
TWO_QUBIT_ERROR = 0.01
READOUT_ERROR = 0.02
PEC_SAMPLES = 300
PEC_SHOTS_PER_SAMPLE = 256
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

LABELS = ["Raw noisy", "ZNE", "PEC (1-qubit noise only)"]
COLORS = ["#d62728", "#2ca02c", "#1f77b4"]
MARKERS = ["o", "s", "^"]


def _timed(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - t0


def _row(n_steps: int, obs, noise_model, exact: float) -> dict:
    qc = trotter_circuit(T_EVOLUTION, n_steps)

    raw, t_raw = _timed(noisy_expectation, qc, obs, noise_model, shots=SHOTS)
    (zne_val, _), t_zne = _timed(zne_expectation, qc, obs, noise_model, scales=(1, 3, 5), shots=SHOTS, return_raw=True)
    (pec_val, pec_overhead), t_pec = _timed(pec_expectation, qc, obs, noise_model, one_qubit_error=ONE_QUBIT_ERROR,
                                             n_samples=PEC_SAMPLES, shots_per_sample=PEC_SHOTS_PER_SAMPLE, seed=SEED)
    return {
        "n_steps": n_steps, "depth": qc.depth(), "two_q_gates": qc.count_ops().get("rzz", 0), "exact": exact,
        "raw": raw, "raw_err": abs(raw - exact),
        "zne": zne_val, "zne_err": abs(zne_val - exact),
        "pec": pec_val, "pec_err": abs(pec_val - exact), "pec_overhead": pec_overhead,
        "t_raw_sec": t_raw, "t_zne_sec": t_zne, "t_pec_sec": t_pec,
    }


def sweep() -> list:
    obs = magnetization()
    noise_model = build_noise_model(two_qubit_error=TWO_QUBIT_ERROR, one_qubit_error=ONE_QUBIT_ERROR,
                                     readout_error=READOUT_ERROR)
    exact = exact_magnetization(T_EVOLUTION)

    rows = [_row(n, obs, noise_model, exact) for n in STEP_COUNTS]
    for r in rows:
        print(f"n_steps={r['n_steps']:3d} depth={r['depth']:4d}  raw_err={r['raw_err']:.4f}  "
              f"zne_err={r['zne_err']:.4f}  pec_err={r['pec_err']:.4f}  pec_overhead={r['pec_overhead']:.2f}x")
    return rows


def save_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(x, series, xlabel, ylabel, title, path, exact_line=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if exact_line is not None:
        ax.axhline(exact_line, color="black", linestyle="--", label="Exact (diagonalization)")
    for values, label, color, marker in zip(series, LABELS, COLORS, MARKERS):
        ax.plot(x, values, f"{marker}-", label=label, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error(rows: list, path: Path) -> None:
    x = [r["n_steps"] for r in rows]
    series = [[r[k] for r in rows] for k in ("raw_err", "zne_err", "pec_err")]
    _plot(x, series, "Trotter steps (circuit depth proxy)", "|error| vs. exact diagonalization",
          "Mitigation Method Comparison: Absolute Error vs. Trotter Depth", path)


def plot_expectation(rows: list, path: Path) -> None:
    x = [r["n_steps"] for r in rows]
    series = [[r[k] for r in rows] for k in ("raw", "zne", "pec")]
    _plot(x, series, "Trotter steps", "<Magnetization>",
          "Estimated Magnetization vs. Trotter Depth (U-shaped Trotter/noise tradeoff)", path,
          exact_line=rows[0]["exact"])


def plot_zne_example(path: Path, n_steps: int = 16) -> None:
    obs = magnetization()
    noise_model = build_noise_model(two_qubit_error=TWO_QUBIT_ERROR, one_qubit_error=ONE_QUBIT_ERROR,
                                     readout_error=READOUT_ERROR)
    qc = trotter_circuit(T_EVOLUTION, n_steps)
    zne_val, points = zne_expectation(qc, obs, noise_model, scales=(1, 3, 5), shots=SHOTS, return_raw=True)
    scales, measured = zip(*points)

    coeffs = np.polyfit(scales, measured, deg=2)
    xs = np.linspace(0, max(scales), 100)
    ys = np.polyval(coeffs, xs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scales, measured, "o", color="#2ca02c", markersize=10, label="Measured (folded circuits)")
    ax.plot(xs, ys, "--", color="#2ca02c", alpha=0.6, label="Richardson extrapolation fit")
    ax.axvline(0, color="gray", linestyle=":")
    ax.plot(0, zne_val, "*", color="red", markersize=18, label=f"ZNE estimate = {zne_val:.4f}")
    ax.set(xlabel="Noise scale factor (gate-folding multiplier)", ylabel="Measured <Magnetization>",
           title=f"ZNE Extrapolation Example (n_steps={n_steps})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Running raw / ZNE / PEC benchmark sweep...")
    rows = sweep()
    save_csv(rows, RESULTS_DIR / "benchmark_results.csv")
    plot_error(rows, RESULTS_DIR / "error_vs_depth.png")
    plot_expectation(rows, RESULTS_DIR / "expectation_vs_depth.png")
    plot_zne_example(RESULTS_DIR / "zne_extrapolation_example.png")
    print(f"\nDone. Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
