from __future__ import annotations

import os
from typing import Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


PAPER_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

STAGE_ORDER = ["warmup", "joint", "finetune"]
STAGE_LABEL = {"warmup": "Stage 1 (Warmup)", "joint": "Stage 2 (Joint)", "finetune": "Stage 3 (Finetune)"}
CONDITION_LABEL = {"C": "Clean", "S": "Single-SNR", "M": "Mixed-SNR"}
CONDITION_COLOR = {"C": "#1f77b4", "S": "#ff7f0e", "M": "#2ca02c"}


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _setup_style() -> None:
    plt.rcParams.update(PAPER_STYLE)
    sns.set_theme(style="whitegrid")


def _pick_focus_setting(training_history_df: pd.DataFrame) -> tuple[int, str]:
    if training_history_df.empty:
        return 2, "M"
    k_values = sorted(int(v) for v in training_history_df["k"].unique())
    cond_values = set(training_history_df["train_condition"].astype(str).unique())
    focus_k = 2 if 2 in k_values else k_values[0]
    focus_cond = "M" if "M" in cond_values else sorted(cond_values)[0]
    return focus_k, focus_cond


def plot_training_diagnostics(
    pretrain_history_df: pd.DataFrame,
    training_history_df: pd.DataFrame,
    output_path: str,
) -> None:
    if pretrain_history_df.empty and training_history_df.empty:
        return
    _setup_style()
    _ensure_dir(os.path.dirname(output_path))

    focus_k, focus_cond = _pick_focus_setting(training_history_df)
    focus_df = training_history_df[
        (training_history_df["k"] == focus_k) & (training_history_df["train_condition"] == focus_cond)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    if not pretrain_history_df.empty:
        pre = pretrain_history_df.sort_values("step")
        ax_a.plot(pre["step"], pre["loss"], color="#d62728", linewidth=2.0, label="Contrastive loss")
        ax_a.set_xlabel("Pretrain Step")
        ax_a.set_ylabel("Loss")
        ax_a2 = ax_a.twinx()
        ax_a2.plot(pre["step"], pre["lr"], color="#9467bd", linewidth=1.8, alpha=0.8, label="Learning rate")
        ax_a2.set_ylabel("Learning Rate")
        ax_a2.set_yscale("log")
        ax_a.set_title("(a) Contrastive Pretraining")
    else:
        ax_a.text(0.5, 0.5, "No pretraining history", ha="center", va="center")
        ax_a.set_axis_off()

    if not focus_df.empty:
        stage_stats = (
            focus_df.groupby(["stage", "epoch"], as_index=False)
            .agg(loss_mean=("loss", "mean"), loss_std=("loss", "std"))
            .fillna(0.0)
        )
        offset = 0
        for stage in STAGE_ORDER:
            sub = stage_stats[stage_stats["stage"] == stage].sort_values("epoch")
            if sub.empty:
                continue
            x = np.arange(1, len(sub) + 1) + offset
            color = {"warmup": "#4C72B0", "joint": "#55A868", "finetune": "#C44E52"}[stage]
            ax_b.plot(x, sub["loss_mean"], color=color, linewidth=2, label=STAGE_LABEL[stage])
            ax_b.fill_between(
                x,
                sub["loss_mean"] - sub["loss_std"],
                sub["loss_mean"] + sub["loss_std"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            if offset > 0:
                ax_b.axvline(offset + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            offset += len(sub)
        ax_b.set_xlabel("Epoch (Concatenated Across Stages)")
        ax_b.set_ylabel("Training Loss")
        ax_b.set_title(f"(b) Three-Stage Loss (k={focus_k}, {CONDITION_LABEL.get(focus_cond, focus_cond)})")
        ax_b.legend(frameon=True)
    else:
        ax_b.text(0.5, 0.5, "No stage history", ha="center", va="center")
        ax_b.set_axis_off()

    if not focus_df.empty and focus_df["alpha"].notna().any():
        alpha_stats = (
            focus_df[focus_df["alpha"].notna()]
            .groupby(["stage", "epoch"], as_index=False)
            .agg(alpha_mean=("alpha", "mean"), alpha_std=("alpha", "std"))
            .fillna(0.0)
        )
        offset = 0
        for stage in STAGE_ORDER:
            sub = alpha_stats[alpha_stats["stage"] == stage].sort_values("epoch")
            if sub.empty:
                continue
            x = np.arange(1, len(sub) + 1) + offset
            color = {"warmup": "#4C72B0", "joint": "#55A868", "finetune": "#C44E52"}[stage]
            ax_c.plot(x, sub["alpha_mean"], color=color, linewidth=2, label=STAGE_LABEL[stage])
            ax_c.fill_between(
                x,
                sub["alpha_mean"] - sub["alpha_std"],
                sub["alpha_mean"] + sub["alpha_std"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            offset += len(sub)
        ax_c.set_xlabel("Epoch (Concatenated Across Stages)")
        ax_c.set_ylabel("Alpha")
        ax_c.set_ylim(0.0, 1.0)
        ax_c.set_title("(c) Learnable Residual Weight")
        ax_c.legend(frameon=True)
    else:
        ax_c.text(0.5, 0.5, "No alpha trajectory", ha="center", va="center")
        ax_c.set_axis_off()

    if not training_history_df.empty:
        last_epoch_rows = (
            training_history_df.sort_values(["seed", "k", "train_condition", "stage", "epoch"])
            .groupby(["seed", "k", "train_condition", "stage"], as_index=False)
            .tail(1)
        )
        sns.boxplot(
            data=last_epoch_rows,
            x="stage",
            y="loss",
            hue="train_condition",
            order=STAGE_ORDER,
            palette=CONDITION_COLOR,
            ax=ax_d,
        )
        ax_d.set_xlabel("Stage")
        ax_d.set_ylabel("Final Epoch Loss")
        ax_d.set_title("(d) Final Stage Loss Distribution")
        ax_d.legend(title="Condition", frameon=True)
    else:
        ax_d.text(0.5, 0.5, "No loss distribution", ha="center", va="center")
        ax_d.set_axis_off()

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gate_activation_distribution(gate_samples_df: pd.DataFrame, output_path: str, focus_k: int = 2) -> None:
    if gate_samples_df.empty:
        return
    _setup_style()
    _ensure_dir(os.path.dirname(output_path))

    sub = gate_samples_df[gate_samples_df["k"] == focus_k].copy()
    if sub.empty:
        sub = gate_samples_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for cond in sorted(sub["train_condition"].unique()):
        cond_df = sub[sub["train_condition"] == cond]
        sns.kdeplot(
            data=cond_df,
            x="gate_activation",
            fill=True,
            alpha=0.25,
            linewidth=1.8,
            color=CONDITION_COLOR.get(cond, "#333333"),
            label=CONDITION_LABEL.get(cond, str(cond)),
            ax=axes[0],
        )
    axes[0].set_title(f"(a) Gate Activation Density (k={focus_k})")
    axes[0].set_xlabel("Gate Activation")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    summary = (
        sub.groupby(["k", "train_condition"], as_index=False)
        .agg(gate_mean=("gate_activation", "mean"), gate_std=("gate_activation", "std"))
        .fillna(0.0)
    )
    summary = summary.sort_values("train_condition").reset_index(drop=True)
    x = np.arange(len(summary))
    bar_colors = [CONDITION_COLOR.get(c, "#333333") for c in summary["train_condition"]]
    axes[1].bar(x, summary["gate_mean"], color=bar_colors, alpha=0.85)
    axes[1].errorbar(
        x=x,
        y=summary["gate_mean"],
        yerr=summary["gate_std"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CONDITION_LABEL.get(c, c) for c in summary["train_condition"]])
    axes[1].set_xlabel("Train Condition")
    axes[1].set_ylabel("Gate Activation Mean +/- Std")
    axes[1].set_title("(b) Gate Activation Summary")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _class_order(class_names: Iterable[str], in_df: pd.DataFrame) -> list[str]:
    names = [str(c) for c in class_names] if class_names else []
    if names:
        return names
    if "class_name" in in_df.columns:
        return sorted(str(v) for v in in_df["class_name"].unique())
    return []


def plot_per_class_metric_grid(
    per_class_summary_df: pd.DataFrame,
    class_names: Iterable[str],
    output_path: str,
    metric_prefix: str = "ap",
    focus_condition: str = "M",
) -> None:
    if per_class_summary_df.empty:
        return
    _setup_style()
    _ensure_dir(os.path.dirname(output_path))

    sub = per_class_summary_df[per_class_summary_df["train_condition"] == focus_condition].copy()
    if sub.empty:
        sub = per_class_summary_df.copy()

    k_values = sorted(int(v) for v in sub["k"].unique())
    class_order = _class_order(class_names, sub)
    snr_order = sorted(float(v) for v in sub["test_snr"].unique())
    if not k_values or not class_order or not snr_order:
        return

    n_cols = min(3, len(k_values))
    n_rows = int(np.ceil(len(k_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)
    value_col = f"{metric_prefix}_mean"

    for idx, k in enumerate(k_values):
        ax = cast(Axes, axes[idx // n_cols, idx % n_cols])
        k_sub = sub[sub["k"] == k]
        pivot = (
            k_sub.pivot_table(index="class_name", columns="test_snr", values=value_col, aggfunc="mean")
            .reindex(index=class_order, columns=snr_order)
        )
        sns.heatmap(
            pivot,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel("Test SNR (dB)")
        ax.set_ylabel("Species")

    for idx in range(len(k_values), n_rows * n_cols):
        cast(Axes, axes[idx // n_cols, idx % n_cols]).axis("off")

    metric_name = "AP" if metric_prefix == "ap" else "AUC"
    fig.suptitle(f"Per-Species {metric_name} Heatmaps ({CONDITION_LABEL.get(focus_condition, focus_condition)} Training)", y=0.995)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_species_error_analysis(
    per_class_summary_df: pd.DataFrame,
    output_path: str,
    focus_k: int = 2,
    focus_snr: float = -5.0,
    focus_condition: str = "M",
) -> None:
    if per_class_summary_df.empty:
        return
    _setup_style()
    _ensure_dir(os.path.dirname(output_path))

    sub = per_class_summary_df[
        (per_class_summary_df["k"] == focus_k)
        & (per_class_summary_df["test_snr"] == focus_snr)
        & (per_class_summary_df["train_condition"] == focus_condition)
    ].copy()
    if sub.empty:
        return
    sub = sub.sort_values("ap_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(sub["class_name"], sub["ap_mean"], color="#1f77b4", alpha=0.85)
    ax.errorbar(
        sub["ap_mean"],
        np.arange(len(sub)),
        xerr=sub["ap_std"].fillna(0.0),
        fmt="none",
        ecolor="black",
        capsize=3,
        elinewidth=1.1,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Average Precision (Mean +/- Std)")
    ax.set_ylabel("Species")
    ax.set_title(f"Species-Level Error Analysis (k={focus_k}, test SNR={focus_snr:g}dB)")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_paper_visualizations(
    results_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    training_history_df: pd.DataFrame,
    gate_samples_df: pd.DataFrame,
    pretrain_history_df: pd.DataFrame,
    class_names: Iterable[str],
    output_dir: str,
) -> None:
    if (
        results_df.empty
        and per_class_df.empty
        and training_history_df.empty
        and gate_samples_df.empty
        and pretrain_history_df.empty
    ):
        return

    _ensure_dir(output_dir)
    paper_dir = os.path.join(output_dir, "paper_figures")
    _ensure_dir(paper_dir)

    plot_training_diagnostics(
        pretrain_history_df=pretrain_history_df,
        training_history_df=training_history_df,
        output_path=os.path.join(paper_dir, "fig_training_diagnostics.png"),
    )
    plot_gate_activation_distribution(
        gate_samples_df=gate_samples_df,
        output_path=os.path.join(paper_dir, "fig_gate_activation_distribution.png"),
        focus_k=2,
    )

    if not per_class_df.empty:
        per_class_summary = (
            per_class_df.groupby(["k", "train_condition", "test_snr", "class_idx", "class_name"], as_index=False)
            .agg(
                ap_mean=("ap", "mean"),
                ap_std=("ap", "std"),
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
            )
            .fillna(0.0)
        )
        plot_per_class_metric_grid(
            per_class_summary_df=per_class_summary,
            class_names=class_names,
            output_path=os.path.join(paper_dir, "fig_per_species_ap_heatmap.png"),
            metric_prefix="ap",
            focus_condition="M",
        )
        plot_per_class_metric_grid(
            per_class_summary_df=per_class_summary,
            class_names=class_names,
            output_path=os.path.join(paper_dir, "fig_per_species_auc_heatmap.png"),
            metric_prefix="auc",
            focus_condition="M",
        )
        plot_species_error_analysis(
            per_class_summary_df=per_class_summary,
            output_path=os.path.join(paper_dir, "fig_species_error_analysis.png"),
            focus_k=2,
            focus_snr=-5.0,
            focus_condition="M",
        )
