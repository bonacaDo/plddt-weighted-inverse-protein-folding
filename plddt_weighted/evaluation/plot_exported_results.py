#!/usr/bin/env python3
"""
plot_exported_results.py
========================
Create presentation-ready plots from exported aggregate evaluation results.

This script is meant for the real exported results under:
    figures/exported_results/

It uses:
    pLDDT_results  - pLDDT-weighted model run

Outputs:
    - overall recovery comparison
    - overall gap to vanilla
    - very_low / low improvement comparison
    - new model recovery by pLDDT bin
    - focused low-bin summary
    - training curve
"""

import argparse
import json
import re
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BENCHMARKS = [
    ("large_mixed", "Large\nmixed"),
    ("lowconf_set", "Low-conf\nenriched"),
    ("strict_lowconf", "Strict\nlow-conf"),
]

MODEL_KEY = "pLDDT_results"

MODEL_DIRS = [
    (MODEL_KEY, "pLDDT model", "#D95F43"),
]

BIN_ORDER = ["very_low", "low", "confident", "very_high"]
BIN_LABELS = {
    "very_low": "Very low\n<50",
    "low": "Low\n50-70",
    "confident": "Confident\n70-90",
    "very_high": "Very high\n>90",
}

COLORS = {
    "vanilla": "#3D7DBD",
    "old": "#6F7D8C",
    "new": "#D95F43",
    "positive": "#2E9D57",
    "negative": "#B23B3B",
    "grid": "#D8D6CF",
    "text": "#24313F",
    "bg": "#FBFAF5",
}


def percent_improvement(vanilla, plddt):
    if vanilla == 0:
        return 0.0
    return (plddt - vanilla) / vanilla * 100.0


def format_percent(value):
    return f"{value:+.2f}%"


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_aggregates(export_dir):
    data = {}
    for model_dir, _, _ in MODEL_DIRS:
        model_data = {}
        canonical_path = export_dir / model_dir / "pLDDT_results_aggregate.json"
        if canonical_path.exists():
            model_data["pLDDT_results"] = load_json(canonical_path)

        for bench_key, _ in BENCHMARKS:
            path = export_dir / model_dir / f"{bench_key}_aggregate.json"
            if path.exists():
                model_data[bench_key] = load_json(path)
        # Fallback: direct output from evaluate_models.py (single run)
        if not model_data:
            direct_path = export_dir / "aggregate_results.json"
            if direct_path.exists():
                model_data["evaluation"] = load_json(direct_path)
        data[model_dir] = model_data

    focus_path = export_dir / MODEL_KEY / "focus_lowbins_aggregate.json"
    focus_data = load_json(focus_path) if focus_path.exists() else None

    return data, focus_data


def load_targeted_results(export_dir):
    target_dir = export_dir / "targeted_lowconf_wins"
    required = {
        "large_very_low_only": target_dir / "large_very_low_only_aggregate.json",
        "known_win": target_dir / "known_win_aggregate.json",
        "candidate_lowconf_win": target_dir / "candidate_lowconf_win_aggregate.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing targeted result files:\n  " + "\n  ".join(missing)
        )

    data = {key: load_json(path) for key, path in required.items()}
    logs = {
        "known_win": target_dir / "eval_plddt_known_win_66184695.txt",
        "candidate_lowconf_win": target_dir / "eval_plddt_candidate_lowconf_win_focus_66184700.txt",
        "large_very_low_only": target_dir / "eval_plddt_large_very_low_only_66184689.txt",
    }
    return data, logs


def parse_eval_log(path):
    if not path.exists():
        return []

    pattern = re.compile(
        r"Evaluating (?P<name>AF-[A-Z0-9]+-F1\.pdb)\.\.\. "
        r"vanilla=(?P<vanilla>[\d.]+), pLDDT=(?P<plddt>[\d.]+), "
        r"delta=(?P<delta>[+-][\d.]+)"
    )
    rows = []
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append({
            "name": item["name"],
            "vanilla": float(item["vanilla"]),
            "plddt": float(item["plddt"]),
            "delta": float(item["delta"]),
        })
    return rows


def parse_training_log(path):
    if not path.exists():
        return []

    pattern = re.compile(
        r"epoch: (?P<epoch>\d+), step: (?P<step>\d+), time: (?P<time>[\d.]+), "
        r"train: (?P<train>[\d.]+), valid: (?P<valid>[\d.]+), "
        r"train_acc: (?P<train_acc>[\d.]+), valid_acc: (?P<valid_acc>[\d.]+)"
    )

    rows = []
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        row = {
            key: float(value) if key not in {"epoch", "step"} else int(value)
            for key, value in match.groupdict().items()
        }
        rows.append(row)
    return rows


def style_axes(ax):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=COLORS["text"])
    ax.title.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_new_model_bins(data, output_dir):
    bench = data[MODEL_KEY]["large_mixed"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor(COLORS["bg"])

    x = list(range(len(BIN_ORDER)))
    width = 0.36
    vanilla = [bench[f"{bin_name}_vanilla_recovery"] for bin_name in BIN_ORDER]
    plddt = [bench[f"{bin_name}_plddt_recovery"] for bin_name in BIN_ORDER]

    ax.bar([v - width / 2 for v in x], vanilla, width, label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width, label="pLDDT model", color=COLORS["new"])

    for xpos, v, p in zip(x, vanilla, plddt):
        delta = percent_improvement(v, p)
        color = COLORS["positive"] if delta > 0 else COLORS["negative"]
        ax.text(xpos, max(v, p) + 0.015, format_percent(delta), ha="center",
                color=color, fontsize=10, weight="bold")

    ax.set_title("pLDDT-Bin Recovery",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Mean sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[name] for name in BIN_ORDER])
    ax.set_ylim(0, max(vanilla + plddt) * 1.18)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "new_model_recovery_by_plddt_bin")


def plot_focus_lowbins(focus_data, output_dir):
    if not focus_data:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])

    labels = ["Combined\nvery_low + low", "Very low\n<50", "Low\n50-70"]
    vanilla = [
        focus_data["focus_vanilla_recovery"],
        focus_data["focus_per_bin"]["very_low"]["vanilla_recovery"],
        focus_data["focus_per_bin"]["low"]["vanilla_recovery"],
    ]
    plddt = [
        focus_data["focus_plddt_recovery"],
        focus_data["focus_per_bin"]["very_low"]["plddt_recovery"],
        focus_data["focus_per_bin"]["low"]["plddt_recovery"],
    ]

    x = list(range(len(labels)))
    width = 0.34
    ax.bar([v - width / 2 for v in x], vanilla, width, label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width, label="pLDDT model", color=COLORS["new"])

    for xpos, v, p in zip(x, vanilla, plddt):
        delta = percent_improvement(v, p)
        color = COLORS["positive"] if delta > 0 else COLORS["negative"]
        ax.text(xpos, max(v, p) + 0.008, format_percent(delta), ha="center",
                color=color, fontsize=10, weight="bold")

    ax.set_title("Residue-Weighted Focused Low-Confidence Summary",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(vanilla + plddt) * 1.25)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "focused_lowbins_residue_weighted")


def plot_new_only_overall(data, output_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.patch.set_facecolor(COLORS["bg"])

    x = list(range(len(BENCHMARKS)))
    width = 0.34
    vanilla = []
    plddt = []
    for bench_key, _ in BENCHMARKS:
        bench = data[MODEL_KEY][bench_key]
        vanilla.append(bench["vanilla_mean_recovery"])
        plddt.append(bench["plddt_mean_recovery"])

    ax.bar([v - width / 2 for v in x], vanilla, width,
           label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width,
           label="pLDDT model", color=COLORS["new"])

    for xpos, v, p in zip(x, vanilla, plddt):
        delta = percent_improvement(v, p)
        color = COLORS["positive"] if delta > 0 else COLORS["negative"]
        ax.text(xpos, max(v, p) + 0.015, format_percent(delta),
                ha="center", color=color, fontsize=10, weight="bold")

    ax.set_title("Low-Confidence-Focused Model vs Vanilla",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Mean sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in BENCHMARKS])
    ax.set_ylim(0, max(vanilla + plddt) * 1.18)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "lowconf_focus_vs_vanilla_overall")


def plot_new_only_improvements(data, output_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.patch.set_facecolor(COLORS["bg"])

    labels = []
    values = []
    for bench_key, bench_label in BENCHMARKS:
        bench = data[MODEL_KEY][bench_key]
        labels.append(bench_label)
        values.append(percent_improvement(bench["vanilla_mean_recovery"], bench["plddt_mean_recovery"]))

    x = list(range(len(labels)))
    colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in values]
    ax.axhline(0, color=COLORS["text"], linewidth=1.0, alpha=0.6)
    ax.bar(x, values, color=colors, width=0.52)

    for xpos, val in zip(x, values):
        ax.text(xpos, val + (0.6 if val >= 0 else -0.9), format_percent(val),
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, weight="bold")

    ax.set_title("Overall Improvement vs Vanilla (%)", fontsize=15, weight="bold")
    ax.set_ylabel("Recovery improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax)
    save_figure(fig, output_dir, "lowconf_focus_gap_to_vanilla")


def plot_new_only_low_bins(data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    fig.patch.set_facecolor(COLORS["bg"])
    x = list(range(len(BENCHMARKS)))

    for ax, bin_name in zip(axes, ["very_low", "low"]):
        values = [
            percent_improvement(
                data[MODEL_KEY][bench_key][f"{bin_name}_vanilla_recovery"],
                data[MODEL_KEY][bench_key][f"{bin_name}_plddt_recovery"],
            )
            for bench_key, _ in BENCHMARKS
        ]
        colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in values]
        ax.axhline(0, color=COLORS["text"], linewidth=1.0, alpha=0.6)
        ax.bar(x, values, color=colors, width=0.52)
        for xpos, val in zip(x, values):
            ax.text(xpos, val + (0.6 if val >= 0 else -0.9), format_percent(val),
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, weight="bold")
        ax.set_title(BIN_LABELS[bin_name].replace("\n", " "),
                     fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in BENCHMARKS])
        style_axes(ax)

    axes[0].set_ylabel("Recovery improvement (%)")
    fig.suptitle("Target Low-Confidence Bin Improvements (%)",
                 fontsize=16, weight="bold", color=COLORS["text"])
    save_figure(fig, output_dir, "lowconf_focus_low_bin_improvements")


def plot_new_only_bin_recovery_by_benchmark(data, output_dir):
    model_data = data.get(MODEL_KEY, {})
    if "large_mixed" in model_data:
        bench = model_data["large_mixed"]
    elif model_data:
        # Single-run fallback uses the first available aggregate.
        bench = next(iter(model_data.values()))
    else:
        raise SystemExit(
            "No aggregate data found. Provide either figures/exported_results/pLDDT_results/*.json "
            "or results/evaluation_pLDDT_results/aggregate_results.json"
        )
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(COLORS["bg"])
    x = list(range(len(BIN_ORDER)))
    width = 0.35
    vanilla = [bench[f"{bin_name}_vanilla_recovery"] for bin_name in BIN_ORDER]
    plddt = [bench[f"{bin_name}_plddt_recovery"] for bin_name in BIN_ORDER]

    ax.bar([v - width / 2 for v in x], vanilla, width,
        label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width,
        label="pLDDT model", color=COLORS["new"])

    for xpos, v, p in zip(x, vanilla, plddt):
     delta = percent_improvement(v, p)
     color = COLORS["positive"] if delta > 0 else COLORS["negative"]
     ax.text(xpos, max(v, p) + 0.012, format_percent(delta),
          ha="center", color=color, fontsize=9, weight="bold")

    ax.set_title("pLDDT-Bin Recovery", fontsize=15, weight="bold")
    ax.set_ylabel("Mean sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[name] for name in BIN_ORDER])
    ax.set_ylim(0, max(vanilla + plddt) * 1.18)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "pLDDT_results_by_bin")


def write_new_only_summary(data, focus_data, training_rows, output_dir):
    lines = [
        "# Low-Confidence-Focused Model Plots",
        "",
        "These plots compare only the low-confidence-focused model against vanilla ProteinMPNN.",
        "",
        "## Key Numbers",
        "",
    ]

    for bench_key, label in BENCHMARKS:
        pretty = label.replace("\n", " ")
        bench = data[MODEL_KEY][bench_key]
        lines.append(
            f"- {pretty}: overall gap {format_percent(percent_improvement(bench['vanilla_mean_recovery'], bench['plddt_mean_recovery']))}, "
            f"very_low {format_percent(percent_improvement(bench['very_low_vanilla_recovery'], bench['very_low_plddt_recovery']))}, "
            f"low {format_percent(percent_improvement(bench['low_vanilla_recovery'], bench['low_plddt_recovery']))}"
        )

    if focus_data:
        lines.extend([
            "",
            "## Residue-Weighted Focused Low-Bin Summary",
            "",
            f"- Focus residues: {focus_data['focus_n_residues']}",
            f"- Combined very_low + low improvement: {format_percent(percent_improvement(focus_data['focus_vanilla_recovery'], focus_data['focus_plddt_recovery']))}",
            f"- very_low improvement: {format_percent(percent_improvement(focus_data['focus_per_bin']['very_low']['vanilla_recovery'], focus_data['focus_per_bin']['very_low']['plddt_recovery']))}",
            f"- low improvement: {format_percent(percent_improvement(focus_data['focus_per_bin']['low']['vanilla_recovery'], focus_data['focus_per_bin']['low']['plddt_recovery']))}",
        ])

    if training_rows:
        last = training_rows[-1]
        lines.extend([
            "",
            "## Final Training Epoch",
            "",
            f"- Epoch: {last['epoch']}",
            f"- Train perplexity: {last['train']:.3f}",
            f"- Validation perplexity: {last['valid']:.3f}",
            f"- Train accuracy: {last['train_acc']:.3f}",
            f"- Validation accuracy: {last['valid_acc']:.3f}",
        ])

    lines.extend([
        "",
        "## Generated Files",
        "",
        "- `lowconf_focus_vs_vanilla_overall.png` / `.svg`",
        "- `lowconf_focus_gap_to_vanilla.png` / `.svg`",
        "- `lowconf_focus_low_bin_improvements.png` / `.svg`",
        "- `pLDDT_results_by_bin.png` / `.svg`",
        "- `lowconf_focus_bins_lowconf_set.png` / `.svg`",
        "- `lowconf_focus_bins_strict_lowconf.png` / `.svg`",
        "- `focused_lowbins_residue_weighted.png` / `.svg`",
        "- `training_curve.png` / `.svg`",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def make_new_only_plots(data, focus_data, training_rows, output_dir):
    plot_new_only_bin_recovery_by_benchmark(data, output_dir)


def plot_targeted_overall(targeted_data, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.patch.set_facecolor(COLORS["bg"])

    items = [
        ("known_win", "Known-win\nsubset"),
        ("candidate_lowconf_win", "Candidate\nlow-conf set"),
    ]
    x = list(range(len(items)))
    width = 0.34
    vanilla = [targeted_data[key]["vanilla_mean_recovery"] for key, _ in items]
    plddt = [targeted_data[key]["plddt_mean_recovery"] for key, _ in items]

    ax.bar([v - width / 2 for v in x], vanilla, width,
           label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width,
           label="pLDDT model", color=COLORS["new"])

    for xpos, v, p in zip(x, vanilla, plddt):
         delta = percent_improvement(v, p)
         ax.text(xpos, max(v, p) + 0.012, format_percent(delta),
                ha="center", color=COLORS["positive"] if delta > 0 else COLORS["negative"],
                fontsize=10, weight="bold")

    ax.set_title("Targeted Sets Where the Model Is Competitive",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Mean sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in items])
    ax.set_ylim(0, max(vanilla + plddt) * 1.22)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "targeted_overall_recovery")


def plot_targeted_focus_improvement(targeted_data, output_dir):
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    fig.patch.set_facecolor(COLORS["bg"])

    large = targeted_data["large_very_low_only"]
    candidate = targeted_data["candidate_lowconf_win"]
    known = targeted_data["known_win"]
    items = [
        (
            "All data\nvery_low only",
            percent_improvement(large["focus_vanilla_recovery"], large["focus_plddt_recovery"]),
        ),
        (
            "Candidate\nvery_low + low",
            percent_improvement(candidate["focus_vanilla_recovery"], candidate["focus_plddt_recovery"]),
        ),
        (
            "Known-win\nvery_low",
            percent_improvement(known["very_low_vanilla_recovery"], known["very_low_plddt_recovery"]),
        ),
        (
            "Known-win\nlow",
            percent_improvement(known["low_vanilla_recovery"], known["low_plddt_recovery"]),
        ),
    ]

    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    x = list(range(len(items)))
    colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in values]

    ax.axhline(0, color=COLORS["text"], linewidth=1.0, alpha=0.6)
    ax.bar(x, values, color=colors, width=0.56)
    for xpos, val in zip(x, values):
        ax.text(xpos, val + (0.4 if val >= 0 else -0.6), format_percent(val),
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, weight="bold")

    ax.set_title("Targeted Low-Confidence Improvements",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Recovery improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax)
    save_figure(fig, output_dir, "targeted_lowconf_improvements")


def plot_candidate_focus_recovery(targeted_data, output_dir):
    focus = targeted_data["candidate_lowconf_win"]
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    fig.patch.set_facecolor(COLORS["bg"])

    labels = ["very_low + low", "very_low", "low"]
    vanilla = [
        focus["focus_vanilla_recovery"],
        focus["focus_per_bin"]["very_low"]["vanilla_recovery"],
        focus["focus_per_bin"]["low"]["vanilla_recovery"],
    ]
    plddt = [
        focus["focus_plddt_recovery"],
        focus["focus_per_bin"]["very_low"]["plddt_recovery"],
        focus["focus_per_bin"]["low"]["plddt_recovery"],
    ]
    x = list(range(len(labels)))
    width = 0.34

    ax.bar([v - width / 2 for v in x], vanilla, width,
           label="Vanilla", color=COLORS["vanilla"])
    ax.bar([v + width / 2 for v in x], plddt, width,
           label="pLDDT model", color=COLORS["new"])
    for xpos, v, p in zip(x, vanilla, plddt):
         delta = percent_improvement(v, p)
         ax.text(xpos, max(v, p) + 0.010, format_percent(delta),
                ha="center", color=COLORS["positive"] if delta > 0 else COLORS["negative"],
                fontsize=10, weight="bold")

    ax.set_title("Candidate Set: Focused Low-Confidence Recovery",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Sequence recovery")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(vanilla + plddt) * 1.22)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir, "candidate_focus_recovery")


def plot_targeted_per_bin(targeted_data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, (key, title) in zip(
        axes,
        [
            ("known_win", "Known-win subset"),
            ("candidate_lowconf_win", "Candidate low-conf set"),
        ],
    ):
        agg = targeted_data[key]
        values = [
            percent_improvement(agg[f"{bin_name}_vanilla_recovery"], agg[f"{bin_name}_plddt_recovery"])
            for bin_name in BIN_ORDER
        ]
        x = list(range(len(BIN_ORDER)))
        colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in values]
        ax.axhline(0, color=COLORS["text"], linewidth=1.0, alpha=0.6)
        ax.bar(x, values, color=colors, width=0.56)
        for xpos, val in zip(x, values):
            ax.text(xpos, val + (0.4 if val >= 0 else -0.6), format_percent(val),
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, weight="bold")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([BIN_LABELS[name] for name in BIN_ORDER])
        style_axes(ax)

    axes[0].set_ylabel("Recovery improvement (%)")
    fig.suptitle("Targeted Evaluation: Per-Bin Improvements (%)",
                 fontsize=16, weight="bold", color=COLORS["text"])
    save_figure(fig, output_dir, "targeted_per_bin_improvements")


def plot_targeted_win_counts(logs, output_dir):
    datasets = [
        ("known_win", "Known-win\nsubset"),
        ("candidate_lowconf_win", "Candidate\nlow-conf set"),
    ]
    counts = []
    for key, _ in datasets:
        rows = parse_eval_log(logs[key])
        wins = sum(1 for row in rows if row["delta"] > 0.0005)
        ties = sum(1 for row in rows if abs(row["delta"]) <= 0.0005)
        losses = sum(1 for row in rows if row["delta"] < -0.0005)
        counts.append((wins, ties, losses))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(COLORS["bg"])
    x = list(range(len(datasets)))
    bottoms = [0] * len(datasets)
    labels = ["Wins", "Ties", "Losses"]
    colors = [COLORS["positive"], "#C9A227", COLORS["negative"]]
    for idx, label in enumerate(labels):
        vals = [row[idx] for row in counts]
        ax.bar(x, vals, bottom=bottoms, label=label, color=colors[idx], width=0.52)
        for xpos, val, bottom in zip(x, vals, bottoms):
            if val:
                ax.text(xpos, bottom + val / 2, str(val), ha="center", va="center",
                        color="white", fontsize=11, weight="bold")
        bottoms = [bottom + val for bottom, val in zip(bottoms, vals)]

    ax.set_title("Per-Protein Outcomes in Targeted Sets",
                 fontsize=15, weight="bold")
    ax.set_ylabel("Number of proteins")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.legend(frameon=False, ncol=3, loc="upper center")
    style_axes(ax)
    save_figure(fig, output_dir, "targeted_win_tie_loss_counts")


def write_targeted_summary(targeted_data, logs, output_dir):
    cand = targeted_data["candidate_lowconf_win"]
    known = targeted_data["known_win"]
    large = targeted_data["large_very_low_only"]
    cand_rows = parse_eval_log(logs["candidate_lowconf_win"])
    known_rows = parse_eval_log(logs["known_win"])

    lines = [
        "# Targeted Low-Confidence Win Plots",
        "",
        "These plots are targeted diagnostics, not broad benchmark summaries.",
        "",
        "## Key Numbers",
        "",
        f"- Candidate set overall improvement: "
        f"{format_percent(percent_improvement(cand['vanilla_mean_recovery'], cand['plddt_mean_recovery']))}",
        f"- Candidate set focused very_low + low improvement: {format_percent(percent_improvement(cand['focus_vanilla_recovery'], cand['focus_plddt_recovery']))}",
        f"- Candidate set very_low focused improvement: "
        f"{format_percent(percent_improvement(cand['focus_per_bin']['very_low']['vanilla_recovery'], cand['focus_per_bin']['very_low']['plddt_recovery']))}",
        f"- Known-win set overall improvement: "
        f"{format_percent(percent_improvement(known['vanilla_mean_recovery'], known['plddt_mean_recovery']))}",
        f"- Known-win set low-bin improvement: {format_percent(percent_improvement(known['low_vanilla_recovery'], known['low_plddt_recovery']))}",
        f"- All-data very_low-only residue-weighted improvement: "
        f"{format_percent(percent_improvement(large['focus_vanilla_recovery'], large['focus_plddt_recovery']))}",
        "",
        "## Per-Protein Outcomes",
        "",
        f"- Candidate set wins/ties/losses: "
        f"{sum(1 for r in cand_rows if r['delta'] > 0.0005)}/"
        f"{sum(1 for r in cand_rows if abs(r['delta']) <= 0.0005)}/"
        f"{sum(1 for r in cand_rows if r['delta'] < -0.0005)}",
        f"- Known-win set wins/ties/losses: "
        f"{sum(1 for r in known_rows if r['delta'] > 0.0005)}/"
        f"{sum(1 for r in known_rows if abs(r['delta']) <= 0.0005)}/"
        f"{sum(1 for r in known_rows if r['delta'] < -0.0005)}",
        "",
        "## Generated Files",
        "",
        "- `targeted_overall_recovery.png` / `.svg`",
        "- `targeted_lowconf_improvements.png` / `.svg`",
        "- `candidate_focus_recovery.png` / `.svg`",
        "- `targeted_per_bin_improvements.png` / `.svg`",
        "- `targeted_win_tie_loss_counts.png` / `.svg`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def make_targeted_plots(export_dir, output_dir):
    targeted_data, logs = load_targeted_results(export_dir)
    plot_targeted_overall(targeted_data, output_dir)
    plot_targeted_focus_improvement(targeted_data, output_dir)
    plot_candidate_focus_recovery(targeted_data, output_dir)
    plot_targeted_per_bin(targeted_data, output_dir)
    plot_targeted_win_counts(logs, output_dir)


def plot_training_curve(training_rows, output_dir):
    if not training_rows:
        return

    epochs = [row["epoch"] for row in training_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor(COLORS["bg"])

    axes[0].plot(epochs, [row["train"] for row in training_rows],
                 color=COLORS["new"], label="Train perplexity", linewidth=2)
    axes[0].plot(epochs, [row["valid"] for row in training_rows],
                 color=COLORS["vanilla"], label="Validation perplexity", linewidth=2)
    axes[0].set_title("Training Perplexity", fontsize=13, weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Perplexity")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_acc"] for row in training_rows],
                 color=COLORS["new"], label="Train accuracy", linewidth=2)
    axes[1].plot(epochs, [row["valid_acc"] for row in training_rows],
                 color=COLORS["vanilla"], label="Validation accuracy", linewidth=2)
    axes[1].set_title("Training Accuracy", fontsize=13, weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(frameon=False)

    for ax in axes:
        style_axes(ax)

    fig.suptitle("Low-Confidence-Focused Training Run", fontsize=16,
                 weight="bold", color=COLORS["text"])
    save_figure(fig, output_dir, "training_curve")


def main(args):
    if not HAS_MPL:
        raise SystemExit("matplotlib is required. Install it or run in the HPC venv.")

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, focus_data = load_aggregates(export_dir)
    training_rows = parse_training_log(
        export_dir / MODEL_KEY / "training_log.txt"
    )

    make_new_only_plots(data, focus_data, training_rows, output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_dir",
        default="figures/exported_results",
        help="Directory containing pLDDT_results/ exports.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/generated_pLDDT_results",
        help="Directory where the single PNG/SVG plot will be written.",
    )
    parser.add_argument(
        "--new_only",
        action="store_true",
        help="Deprecated: the script now always generates one plot.",
    )
    parser.add_argument(
        "--targeted_only",
        action="store_true",
        help="Deprecated: the script now always generates one plot.",
    )
    main(parser.parse_args())
