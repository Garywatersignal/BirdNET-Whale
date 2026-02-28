import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


_BASE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
_CONDITION_STYLES = {
    "C": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "CleanTrain + NoisyTest"},
    "S": {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "SoundscapeTrain (Single-SNR)"},
    "M": {"color": "#2ca02c", "linestyle": "-.", "marker": "^", "label": "MixedSoundscapeTrain"},
}


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _train_snr_styles(train_snrs: List[float]) -> Dict[float, Dict[str, str]]:
    styles: Dict[float, Dict[str, str]] = {}
    for i, snr in enumerate(train_snrs):
        styles[float(snr)] = {
            "color": _BASE_COLORS[i % len(_BASE_COLORS)],
            "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
            "marker": _MARKERS[i % len(_MARKERS)],
        }
    return styles


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    required = {"k", "train_snr", "test_snr", "macro_ap", "macro_auc", "accuracy"}
    missing = required.difference(set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {sorted(missing)}")

    summary = (
        results_df.groupby(["k", "train_snr", "test_snr"], as_index=False)
        .agg(
            macro_ap_mean=("macro_ap", "mean"),
            macro_ap_std=("macro_ap", "std"),
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
    )
    # std is NaN when there is only one seed; treat as 0 for plotting CI bands.
    for col in ("macro_ap_std", "macro_auc_std", "accuracy_std"):
        summary[col] = summary[col].fillna(0.0)
    return summary


def _aggregate_results_by_condition(results_df: pd.DataFrame) -> pd.DataFrame:
    required = {"k", "train_condition", "test_snr", "macro_ap", "macro_auc", "accuracy"}
    missing = required.difference(set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {sorted(missing)}")

    summary = (
        results_df.groupby(["k", "train_condition", "test_snr"], as_index=False)
        .agg(
            macro_ap_mean=("macro_ap", "mean"),
            macro_ap_std=("macro_ap", "std"),
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
    )
    for col in ("macro_ap_std", "macro_auc_std", "accuracy_std"):
        summary[col] = summary[col].fillna(0.0)
    return summary


def plot_snr_comparison(summary_df: pd.DataFrame, k_values: List[int], train_snrs: List[float], test_snrs: List[float], output_dir: str) -> None:
    """Multi-SNR k-shot curves (color = test SNR, style = train SNR), modeled after eva2_2_1.py."""
    if summary_df.empty or not k_values or not train_snrs or not test_snrs:
        return

    _ensure_dir(output_dir)
    train_styles = _train_snr_styles(train_snrs)
    cmap = plt.get_cmap("viridis", len(test_snrs))

    metrics: List[Tuple[str, str, str]] = [
        ("macro_auc_mean", "macro_auc_std", "Macro AUC"),
        ("macro_ap_mean", "macro_ap_std", "mAP"),
    ]

    lookup: Dict[Tuple[float, int, float, str], float] = {}
    for row in summary_df.to_dict(orient="records"):
        t_snr = float(row["train_snr"])
        k = int(row["k"])
        s_snr = float(row["test_snr"])
        lookup[(t_snr, k, s_snr, "macro_auc_mean")] = float(row["macro_auc_mean"])
        lookup[(t_snr, k, s_snr, "macro_auc_std")] = float(row["macro_auc_std"])
        lookup[(t_snr, k, s_snr, "macro_ap_mean")] = float(row["macro_ap_mean"])
        lookup[(t_snr, k, s_snr, "macro_ap_std")] = float(row["macro_ap_std"])

    for mean_key, std_key, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.asarray(k_values, dtype=float)
        for test_idx, test_snr in enumerate(test_snrs):
            base_color = cmap(test_idx)
            for train_snr in train_snrs:
                style = train_styles[float(train_snr)]
                y = np.array([lookup.get((float(train_snr), int(k), float(test_snr), mean_key), np.nan) for k in k_values], dtype=float)
                s = np.array([lookup.get((float(train_snr), int(k), float(test_snr), std_key), 0.0) for k in k_values], dtype=float)
                m = np.isfinite(y)
                if not np.any(m):
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    color=base_color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2,
                    markersize=6,
                )
                ax.fill_between(x[m], (y - s)[m], (y + s)[m], color=base_color, alpha=0.12, linewidth=0)

        ax.set_xlabel("k-shot")
        ax.set_ylabel(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(k_values)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_title(f"Multi-SNR k-shot Performance Comparison: {title}")
        ax.grid(True, alpha=0.3)

        test_handles = [
            Line2D([], [], color=cmap(i), linestyle="-", linewidth=3, label=f"Test SNR {snr:g}dB")
            for i, snr in enumerate(test_snrs)
        ]
        train_handles = [
            Line2D(
                [],
                [],
                color="black",
                linestyle=train_styles[float(snr)]["linestyle"],
                marker=train_styles[float(snr)]["marker"],
                linewidth=2,
                markersize=6,
                label=f"Train SNR {snr:g}dB",
            )
            for snr in train_snrs
        ]

        leg1 = ax.legend(handles=test_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False, title="Test SNR")
        ax.add_artist(leg1)
        ax.legend(handles=train_handles, bbox_to_anchor=(1.02, 0.56), loc="upper left", fontsize=8, frameon=False, title="Train SNR")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"multi_snr_kshot_comparison_{mean_key.replace('_mean', '')}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_noise_invariance_analysis(summary_df: pd.DataFrame, k_values: List[int], train_snrs: List[float], test_snrs: List[float], output_dir: str) -> None:
    """Noise invariance analysis (std + mean across test SNRs), modeled after eva2_2_1.py."""
    if summary_df.empty or len(test_snrs) < 2:
        return

    _ensure_dir(output_dir)
    train_styles = _train_snr_styles(train_snrs)

    metrics: List[Tuple[str, str]] = [
        ("macro_auc_mean", "Macro AUC"),
        ("macro_ap_mean", "mAP"),
    ]

    for metric_key, metric_title in metrics:
        variance_data: Dict[float, Dict[int, float]] = {float(s): {} for s in train_snrs}
        mean_data: Dict[float, Dict[int, float]] = {float(s): {} for s in train_snrs}

        for train_snr in train_snrs:
            for k in k_values:
                vals = []
                for test_snr in test_snrs:
                    entries = summary_df[
                        (summary_df["train_snr"] == train_snr)
                        & (summary_df["test_snr"] == test_snr)
                        & (summary_df["k"] == k)
                    ]
                    if not entries.empty:
                        vals.append(float(entries.iloc[0][metric_key]))
                if len(vals) >= 2:
                    variance_data[float(train_snr)][int(k)] = float(np.std(vals))
                    mean_data[float(train_snr)][int(k)] = float(np.mean(vals))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for train_snr in train_snrs:
            d = variance_data.get(float(train_snr), {})
            if not d:
                continue
            ks = sorted(d.keys())
            ys = [d[int(k)] for k in ks]
            style = train_styles[float(train_snr)]
            ax1.plot(
                ks,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=f"Train SNR {train_snr:g}dB",
            )

        ax1.set_xlabel("k-shot")
        ax1.set_ylabel(f"Std Dev of {metric_title} across Test SNRs")
        ax1.set_title("Noise Invariance (lower = better)")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(k_values)
        ax1.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, frameon=False)

        for train_snr in train_snrs:
            d = mean_data.get(float(train_snr), {})
            if not d:
                continue
            ks = sorted(d.keys())
            ys = [d[int(k)] for k in ks]
            style = train_styles[float(train_snr)]
            ax2.plot(
                ks,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=f"Train SNR {train_snr:g}dB",
            )

        ax2.set_xlabel("k-shot")
        ax2.set_ylabel(f"Mean {metric_title} (across Test SNRs)")
        ax2.set_title("Average Performance")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(k_values)
        ax2.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"noise_invariance_analysis_{metric_key.replace('_mean', '')}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_performance_vs_k_and_snr(results_df: pd.DataFrame, output_path: str):
    """
    Backward-compatible entrypoint used by main.py.

    It writes:
    - output_path: a compact 2-panel summary (mean mAP + noise invariance)
    - extra plots under <output_path>/../plots/ (multi-SNR curves + invariance for mAP/AUC)
    """
    if "train_condition" in results_df.columns:
        summary_df = _aggregate_results_by_condition(results_df)
        k_values = sorted(int(v) for v in summary_df["k"].unique())
        test_snrs = sorted(float(v) for v in summary_df["test_snr"].unique())
        conditions = [c for c in ["C", "S", "M"] if c in summary_df["train_condition"].unique()]

        out_dir = os.path.join(os.path.dirname(output_path), "plots")
        _ensure_dir(out_dir)
        cmap = plt.get_cmap("viridis", len(test_snrs))

        metrics: List[Tuple[str, str, str]] = [
            ("macro_auc_mean", "macro_auc_std", "Macro AUC"),
            ("macro_ap_mean", "macro_ap_std", "mAP"),
        ]

        for mean_key, std_key, title in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.asarray(k_values, dtype=float)
            for test_idx, test_snr in enumerate(test_snrs):
                base_color = cmap(test_idx)
                for cond in conditions:
                    style = _CONDITION_STYLES.get(cond, {"linestyle": "-", "marker": "o"})
                    y_vals = []
                    s_vals = []
                    for k in k_values:
                        rows = summary_df[
                            (summary_df["train_condition"] == cond)
                            & (summary_df["test_snr"] == test_snr)
                            & (summary_df["k"] == k)
                        ]
                        if rows.empty:
                            y_vals.append(np.nan)
                            s_vals.append(0.0)
                        else:
                            y_vals.append(float(rows.iloc[0][mean_key]))
                            s_vals.append(float(rows.iloc[0][std_key]))
                    y = np.array(y_vals, dtype=float)
                    s = np.array(s_vals, dtype=float)
                    m = np.isfinite(y)
                    if not np.any(m):
                        continue
                    ax.plot(
                        x[m],
                        y[m],
                        color=base_color,
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        linewidth=2,
                        markersize=6,
                    )
                    ax.fill_between(x[m], (y - s)[m], (y + s)[m], color=base_color, alpha=0.12, linewidth=0)

            ax.set_xlabel("k-shot")
            ax.set_ylabel(title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(k_values)
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
            ax.set_title(f"Condition Comparison: {title}")
            ax.grid(True, alpha=0.3)

            test_handles = [
                Line2D([], [], color=cmap(i), linestyle="-", linewidth=3, label=f"Test SNR {snr:g}dB")
                for i, snr in enumerate(test_snrs)
            ]
            cond_handles = [
                Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=_CONDITION_STYLES.get(cond, {}).get("linestyle", "-"),
                    marker=_CONDITION_STYLES.get(cond, {}).get("marker", "o"),
                    linewidth=2,
                    markersize=6,
                    label=_CONDITION_STYLES.get(cond, {}).get("label", cond),
                )
                for cond in conditions
            ]
            leg1 = ax.legend(handles=test_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False, title="Test SNR")
            ax.add_artist(leg1)
            ax.legend(handles=cond_handles, bbox_to_anchor=(1.02, 0.56), loc="upper left", fontsize=8, frameon=False, title="Train Condition")

            plt.tight_layout()
            out_path = os.path.join(out_dir, f"condition_comparison_{mean_key.replace('_mean', '')}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        avg_rows = (
            summary_df.groupby(["k", "train_condition"], as_index=False)
            .agg(mean_map=("macro_ap_mean", "mean"), std_map=("macro_ap_mean", "std"))
        )
        avg_rows["std_map"] = avg_rows["std_map"].fillna(0.0)

        inv_rows = []
        for cond in conditions:
            for k in k_values:
                vals = summary_df[(summary_df["train_condition"] == cond) & (summary_df["k"] == k)]["macro_ap_mean"].to_numpy(dtype=float)
                if vals.size >= 2:
                    inv_rows.append({"k": k, "train_condition": cond, "invariance_std": float(np.std(vals))})
                else:
                    inv_rows.append({"k": k, "train_condition": cond, "invariance_std": 0.0})
        inv_df = pd.DataFrame(inv_rows)

        for cond in conditions:
            style = _CONDITION_STYLES.get(cond, {"color": "#000", "linestyle": "-", "marker": "o"})
            sub = avg_rows[avg_rows["train_condition"] == cond].sort_values("k")
            axes[0].plot(
                sub["k"],
                sub["mean_map"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=style.get("label", cond),
            )
            axes[0].fill_between(
                sub["k"],
                sub["mean_map"] - sub["std_map"],
                sub["mean_map"] + sub["std_map"],
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )

            inv_sub = inv_df[inv_df["train_condition"] == cond].sort_values("k")
            axes[1].plot(
                inv_sub["k"],
                inv_sub["invariance_std"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=style.get("label", cond),
            )

        axes[0].set_xlabel("k-shot")
        axes[0].set_ylabel("mAP (mean over Test SNRs)")
        axes[0].set_title("Average Performance vs. k")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(k_values)
        axes[0].get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, frameon=False)

        axes[1].set_xlabel("k-shot")
        axes[1].set_ylabel("Std Dev of mAP across Test SNRs")
        axes[1].set_title("Noise Invariance (lower = better)")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(k_values)
        axes[1].get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8, frameon=False)

        plt.tight_layout()
        _ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return

    summary_df = _aggregate_results(results_df)
    k_values = sorted(int(v) for v in summary_df["k"].unique())
    train_snrs = sorted(float(v) for v in summary_df["train_snr"].unique())
    test_snrs = sorted(float(v) for v in summary_df["test_snr"].unique())

    out_dir = os.path.join(os.path.dirname(output_path), "plots")
    plot_snr_comparison(summary_df, k_values, train_snrs, test_snrs, out_dir)
    plot_noise_invariance_analysis(summary_df, k_values, train_snrs, test_snrs, out_dir)

    train_styles = _train_snr_styles(train_snrs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    avg_rows = (
        summary_df.groupby(["k", "train_snr"], as_index=False)
        .agg(mean_map=("macro_ap_mean", "mean"), std_map=("macro_ap_mean", "std"))
    )
    avg_rows["std_map"] = avg_rows["std_map"].fillna(0.0)

    inv_rows = []
    for train_snr in train_snrs:
        for k in k_values:
            vals = summary_df[(summary_df["train_snr"] == train_snr) & (summary_df["k"] == k)]["macro_ap_mean"].to_numpy(dtype=float)
            if vals.size >= 2:
                inv_rows.append({"k": k, "train_snr": train_snr, "invariance_std": float(np.std(vals))})
            else:
                inv_rows.append({"k": k, "train_snr": train_snr, "invariance_std": 0.0})
    inv_df = pd.DataFrame(inv_rows)

    for train_snr in train_snrs:
        style = train_styles[float(train_snr)]
        sub = avg_rows[avg_rows["train_snr"] == train_snr].sort_values("k")
        axes[0].plot(sub["k"], sub["mean_map"], color=style["color"], linestyle=style["linestyle"], marker=style["marker"], linewidth=2, label=f"Train SNR {train_snr:g}dB")
        axes[0].fill_between(sub["k"], sub["mean_map"] - sub["std_map"], sub["mean_map"] + sub["std_map"], color=style["color"], alpha=0.12, linewidth=0)

    axes[0].set_xlabel("k-shot")
    axes[0].set_ylabel("mAP (mean over Test SNRs)")
    axes[0].set_title("Average Performance vs. k")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(k_values)
    axes[0].get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)

    for train_snr in train_snrs:
        style = train_styles[float(train_snr)]
        sub = inv_df[inv_df["train_snr"] == train_snr].sort_values("k")
        axes[1].plot(sub["k"], sub["invariance_std"], color=style["color"], linestyle=style["linestyle"], marker=style["marker"], linewidth=2, label=f"Train SNR {train_snr:g}dB")

    axes[1].set_xlabel("k-shot")
    axes[1].set_ylabel("Std Dev of mAP across Test SNRs")
    axes[1].set_title("Noise Invariance (lower = better)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(k_values)
    axes[1].get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, frameon=False)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
