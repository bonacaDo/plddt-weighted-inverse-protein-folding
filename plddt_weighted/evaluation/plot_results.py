#!/usr/bin/env python3
"""
plot_results.py
===============
Generate publication-quality plots for the hackathon presentation.

Generates:
1. Bar chart: Vanilla vs pLDDT-weighted recovery by confidence bin
2. Scatter: Per-protein recovery comparison
3. pLDDT profile with per-residue accuracy overlay
4. Architecture diagram data (for slides)

Can run with mock data for presentation preparation before training.

Usage:
    python plot_results.py --results_dir results/evaluation --output_dir figures/
    python plot_results.py --mock --output_dir figures/   # Use mock data
"""

import argparse
import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

# Color scheme for presentation
COLORS = {
    'vanilla': '#4A90D9',       # Blue
    'plddt': '#E8553D',         # Red-orange
    'improvement': '#2ECC71',   # Green
    'bg': '#FAFAFA',
    'grid': '#E0E0E0',
    'text': '#2C3E50',
}

PLDDT_BIN_LABELS = {
    'very_low': 'Very Low\n(pLDDT < 50)',
    'low': 'Low\n(50-70)',
    'confident': 'Confident\n(70-90)',
    'very_high': 'Very High\n(> 90)',
}


def generate_mock_data():
    """Generate realistic mock data for presentation preparation."""
    np.random.seed(42)

    mock_results = []
    proteins = [
        ("Lysozyme (P00698)", 129, 95.2),
        ("Insulin (P01308)", 110, 88.5),
        ("Hemoglobin-β (P68871)", 147, 92.1),
        ("p53 (P04637)", 393, 62.3),
        ("NEMO (Q9Y6K9)", 419, 55.8),
        ("BRCA1 (P38398)", 1863, 48.7),
        ("TDP-43 (Q13148)", 414, 68.4),
        ("Tau (P10636)", 758, 35.2),
        ("hnRNP A1 (P09651)", 372, 52.1),
        ("Aerotaxis (P0A9Q1)", 506, 78.9),
    ]

    for name, length, mean_plddt in proteins:
        # Vanilla recovery correlates somewhat with pLDDT
        base_recovery = 0.25 + 0.45 * (mean_plddt / 100)
        vanilla_rec = base_recovery + np.random.normal(0, 0.02)

        # pLDDT model improves more on mixed-confidence proteins
        disorder_fraction = 1 - mean_plddt / 100
        improvement = 0.01 + 0.08 * disorder_fraction + np.random.normal(0, 0.005)
        plddt_rec = vanilla_rec + improvement

        bins = {}
        for low, high, bname in [(0, 50, "very_low"), (50, 70, "low"),
                                  (70, 90, "confident"), (90, 100, "very_high")]:
            # Simulate residues in this bin
            if mean_plddt < 50:
                weights = [0.4, 0.3, 0.2, 0.1]
            elif mean_plddt < 70:
                weights = [0.15, 0.35, 0.35, 0.15]
            elif mean_plddt < 90:
                weights = [0.05, 0.15, 0.5, 0.3]
            else:
                weights = [0.02, 0.08, 0.3, 0.6]

            bin_idx = [(0, 50), (50, 70), (70, 90), (90, 100)].index((low, high))
            n_res = max(1, int(length * weights[bin_idx]))

            # pLDDT model helps most in low-confidence regions
            if bname == "very_low":
                v_rec = 0.15 + np.random.normal(0, 0.02)
                p_rec = v_rec + 0.06 + np.random.normal(0, 0.01)
            elif bname == "low":
                v_rec = 0.28 + np.random.normal(0, 0.02)
                p_rec = v_rec + 0.04 + np.random.normal(0, 0.01)
            elif bname == "confident":
                v_rec = 0.42 + np.random.normal(0, 0.02)
                p_rec = v_rec + 0.02 + np.random.normal(0, 0.01)
            else:
                v_rec = 0.52 + np.random.normal(0, 0.02)
                p_rec = v_rec + 0.01 + np.random.normal(0, 0.005)

            bins[bname] = {
                'n_residues': n_res,
                'vanilla_recovery': float(np.clip(v_rec, 0, 1)),
                'plddt_recovery': float(np.clip(p_rec, 0, 1)),
                'vanilla_perplexity': float(np.exp(-np.log(max(v_rec, 0.01)) * 2)),
                'plddt_perplexity': float(np.exp(-np.log(max(p_rec, 0.01)) * 2)),
            }

        mock_results.append({
            'name': name,
            'length': length,
            'mean_plddt': mean_plddt,
            'vanilla_recovery': float(np.clip(vanilla_rec, 0, 1)),
            'plddt_recovery': float(np.clip(plddt_rec, 0, 1)),
            'vanilla_perplexity': float(np.exp(-np.log(max(vanilla_rec, 0.01)) * 2)),
            'plddt_perplexity': float(np.exp(-np.log(max(plddt_rec, 0.01)) * 2)),
            'bins': bins,
        })

    return mock_results


def plot_recovery_by_bin(results, output_dir):
    """Bar chart: recovery rate by pLDDT confidence bin."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    bin_names = ['very_low', 'low', 'confident', 'very_high']
    vanilla_means, plddt_means = [], []

    for bname in bin_names:
        v_vals = [r['bins'][bname]['vanilla_recovery']
                  for r in results if bname in r['bins']]
        p_vals = [r['bins'][bname]['plddt_recovery']
                  for r in results if bname in r['bins']]
        vanilla_means.append(np.mean(v_vals) if v_vals else 0)
        plddt_means.append(np.mean(p_vals) if p_vals else 0)

    x = np.arange(len(bin_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, vanilla_means, width,
                   label='Vanilla ProteinMPNN', color=COLORS['vanilla'],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, plddt_means, width,
                   label='pLDDT-Weighted (Ours)', color=COLORS['plddt'],
                   edgecolor='white', linewidth=0.5)

    # Add improvement annotations
    for i, (v, p) in enumerate(zip(vanilla_means, plddt_means)):
        if p > v:
            improvement = (p - v) / v * 100 if v > 0 else 0
            ax.annotate(f'+{improvement:.1f}%',
                        xy=(x[i] + width / 2, p),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold',
                        color=COLORS['improvement'])

    ax.set_xlabel('pLDDT Confidence Bin', fontsize=13, color=COLORS['text'])
    ax.set_ylabel('Sequence Recovery Rate', fontsize=13, color=COLORS['text'])
    ax.set_title('Sequence Recovery by Structural Confidence',
                 fontsize=15, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([PLDDT_BIN_LABELS[b] for b in bin_names], fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, max(max(vanilla_means), max(plddt_means)) * 1.2)
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_by_plddt_bin.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'recovery_by_plddt_bin.svg'),
                bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_by_plddt_bin.png/svg")


def plot_protein_comparison(results, output_dir):
    """Scatter plot: per-protein vanilla vs pLDDT recovery."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    vanilla = [r['vanilla_recovery'] for r in results]
    plddt = [r['plddt_recovery'] for r in results]
    mean_plddt_scores = [r['mean_plddt'] for r in results]
    names = [r['name'].replace('.pdb', '') for r in results]

    scatter = ax.scatter(vanilla, plddt, c=mean_plddt_scores, cmap='RdYlGn',
                         s=120, edgecolors='white', linewidth=1, zorder=5,
                         vmin=30, vmax=100)

    # Diagonal line
    lims = [min(min(vanilla), min(plddt)) - 0.02,
            max(max(vanilla), max(plddt)) + 0.02]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, zorder=1)
    ax.fill_between(lims, lims, [lims[1], lims[1]],
                    alpha=0.05, color=COLORS['plddt'], zorder=0)

    # Label points
    for i, name in enumerate(names):
        short_name = name.split('(')[0].strip() if '(' in name else name[:15]
        ax.annotate(short_name, (vanilla[i], plddt[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Mean pLDDT', fontsize=12)

    ax.set_xlabel('Vanilla ProteinMPNN Recovery', fontsize=13, color=COLORS['text'])
    ax.set_ylabel('pLDDT-Weighted Recovery', fontsize=13, color=COLORS['text'])
    ax.set_title('Per-Protein Sequence Recovery Comparison',
                 fontsize=15, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add "pLDDT better" annotation
    ax.text(0.05, 0.92, 'pLDDT-Weighted\nbetter',
            transform=ax.transAxes, fontsize=11, fontstyle='italic',
            color=COLORS['plddt'], alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'protein_comparison_scatter.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'protein_comparison_scatter.svg'),
                bbox_inches='tight')
    plt.close()
    print("  Saved: protein_comparison_scatter.png/svg")


def plot_improvement_vs_disorder(results, output_dir):
    """Show that improvement correlates with disorder content."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    improvements = []
    disorder_fractions = []
    names = []

    for r in results:
        imp = r['plddt_recovery'] - r['vanilla_recovery']
        improvements.append(imp)
        disorder_fractions.append(1 - r['mean_plddt'] / 100)
        names.append(r['name'].replace('.pdb', ''))

    ax.scatter(disorder_fractions, improvements, s=120,
               c=COLORS['plddt'], edgecolors='white', linewidth=1, zorder=5)

    # Trend line
    z = np.polyfit(disorder_fractions, improvements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(disorder_fractions), max(disorder_fractions), 100)
    ax.plot(x_line, p(x_line), '--', color=COLORS['plddt'], alpha=0.5, linewidth=2)

    # Label points
    for i, name in enumerate(names):
        short_name = name.split('(')[0].strip() if '(' in name else name[:15]
        ax.annotate(short_name, (disorder_fractions[i], improvements[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Disorder Fraction (1 - mean pLDDT / 100)',
                  fontsize=13, color=COLORS['text'])
    ax.set_ylabel('Recovery Improvement (pLDDT - Vanilla)',
                  fontsize=13, color=COLORS['text'])
    ax.set_title('Improvement Correlates with Structural Disorder',
                 fontsize=15, fontweight='bold', color=COLORS['text'], pad=15)
    ax.grid(alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Correlation coefficient
    corr = np.corrcoef(disorder_fractions, improvements)[0, 1]
    ax.text(0.95, 0.05, f'r = {corr:.3f}',
            transform=ax.transAxes, fontsize=12, ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_vs_disorder.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'improvement_vs_disorder.svg'),
                bbox_inches='tight')
    plt.close()
    print("  Saved: improvement_vs_disorder.png/svg")


def plot_summary_table(results, output_dir):
    """Create a summary table as an image for slides."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    headers = ['Protein', 'Length', 'Mean pLDDT',
               'Vanilla\nRecovery', 'pLDDT\nRecovery', 'Improvement']

    cell_data = []
    for r in results:
        name = r['name'].replace('.pdb', '')
        if len(name) > 25:
            name = name[:22] + '...'
        imp = r['plddt_recovery'] - r['vanilla_recovery']
        cell_data.append([
            name,
            str(r['length']),
            f"{r['mean_plddt']:.1f}",
            f"{r['vanilla_recovery']:.3f}",
            f"{r['plddt_recovery']:.3f}",
            f"{imp:+.3f}",
        ])

    table = ax.table(cellText=cell_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor(COLORS['vanilla'])
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color improvement column
    last_col = len(headers) - 1
    for i in range(len(cell_data)):
        imp = float(cell_data[i][-1])
        if imp > 0:
            table[i + 1, last_col].set_facecolor('#E8F8F5')
        else:
            table[i + 1, last_col].set_facecolor('#FDEDEC')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_table.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: results_table.png")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mock:
        print("Using mock data for presentation preparation")
        results = generate_mock_data()
        # Save mock data
        with open(os.path.join(args.output_dir, 'mock_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        results_file = os.path.join(args.results_dir, 'evaluation_results.json')
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            print("Use --mock to generate plots with mock data")
            return
        with open(results_file, 'r') as f:
            results = json.load(f)

    print(f"Generating plots for {len(results)} proteins...")
    plot_recovery_by_bin(results, args.output_dir)
    plot_protein_comparison(results, args.output_dir)
    plot_improvement_vs_disorder(results, args.output_dir)
    plot_summary_table(results, args.output_dir)
    print("\nAll plots generated!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/evaluation')
    parser.add_argument('--output_dir', type=str, default='figures/')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock data for presentation preparation')
    args = parser.parse_args()
    main(args)
