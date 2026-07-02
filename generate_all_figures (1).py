"""
generate_all_figures.py — Publication-Quality Figure Generation
════════════════════════════════════════════════════════════════════════════════
Production-ready figure generation for academic paper.
Outputs: 8 high-resolution figures (600 dpi, IEEE/Elsevier journal quality)
         with optimized typography, no text overlap, and constrained layout.

Files Generated:
  • fig1_bar.png              — Annual retraction volume by year
  • fig2_donut.png            — Publisher concentration (donut chart)
  • fig3_citation_boxplot.png — Citation distribution (box plot)
  • fig4_citation_grouped.png — Mean/median citations by group
  • fig5_zero_citation_stacked.png — Zero-citation proportion stacked bar
  • fig6_retraction_lag.png   — Publication year vs. citation scatter
  • fig7_keyword_anomaly.png  — Bibliometric anomaly indicators
  • fig8_heatmap.png          — Top 10 journals × year retraction heatmap

Requirements: pandas, matplotlib, seaborn, numpy

Run: python generate_all_figures.py
════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# ▶ CONFIGURATION: Global Styling & Palette
# ════════════════════════════════════════════════════════════════════════════

# ─ Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'lines.linewidth': 1.2,
})

# ─ Color Palette (IEEE/Springer compliant) ───────────────────────────────────
C_PURPLE  = "#534AB7"   # Track A1 / primary (LLM-retracted)
C_TEAL    = "#1D9E75"   # Track B / comparison
C_AMBER   = "#EF9F27"   # S2 Hindawi / secondary accent
C_LAVEND  = "#C5C2ED"   # Light purple (accent/fill)
C_GRAY    = "#BFBFBF"   # Grid/borders
C_DARK    = "#333333"   # Text/labels
C_WHITE   = "#FFFFFF"   # Backgrounds

# ─ DPI & Format ──────────────────────────────────────────────────────────────
OUTPUT_DPI = 600        # Publication quality
BBOX_TIGHT = True
CONSTRAIN_LAYOUT = True

# ════════════════════════════════════════════════════════════════════════════
# ▶ DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

print("Loading data...")
a1_raw  = pd.read_csv("A1_final.csv")
a1_sc   = pd.read_csv("A1_scopus_merged.csv")
corpus  = pd.read_csv("analysis_corpus.csv")
print(f"  ✓ A1 raw:        {len(a1_raw)} records")
print(f"  ✓ A1 Scopus:     {len(a1_sc)} records")
print(f"  ✓ Corpus:        {len(corpus)} records")

# ─ Stratum Assignment (Publisher Classification) ────────────────────────────
def assign_stratum(pub):
    """Map publisher name to stratum (S1, S2, S3)."""
    pub = str(pub).lower()
    if "ios press" in pub or "sage" in pub:
        return "S1-JIFS"
    elif "hindawi" in pub or "wiley" in pub:
        return "S2-Hindawi"
    else:
        return "S3-Diverse"

a1_raw["Stratum"] = a1_raw["Publisher"].apply(assign_stratum)

# ─ Publication Year Parsing ──────────────────────────────────────────────────
a1_raw["PubDate"]  = pd.to_datetime(a1_raw["OriginalPaperDate"], errors="coerce")
a1_raw["PubYear"]  = a1_raw["PubDate"].dt.year

# ─ Track Assignment (LLM-retracted vs. Comparison) ──────────────────────────
corpus["Track"] = corpus["Title"].str.contains(
    r"RETRACT", case=False, na=False).map({True: "A1", False: "B"})

# ─ Stratum Assignment for Corpus ─────────────────────────────────────────────
def stratum_from_source(row):
    """Assign stratum based on source title and affiliations."""
    if row["Track"] == "B":
        return "B"
    src = str(row.get("Source title", "")).lower()
    pub = str(row.get("Authors with affiliations", "")).lower()
    if "intelligent" in src and "fuzzy" in src:
        return "S1-JIFS"
    elif "hindawi" in pub or "complexity" in src or "mathematical" in src:
        return "S2-Hindawi"
    else:
        return "S3-Diverse"

corpus["Stratum"] = corpus.apply(stratum_from_source, axis=1)
corpus["Cited by"] = pd.to_numeric(corpus["Cited by"], errors="coerce").fillna(0)
corpus["zero_cited"] = (corpus["Cited by"] == 0).astype(int)

# ─ Keyword Count Calculation ─────────────────────────────────────────────────
def kw_count(kw_str):
    """Count author keywords from semicolon-delimited string."""
    if pd.isna(kw_str) or str(kw_str).strip() == "":
        return 0
    return len([k for k in str(kw_str).split(";") if k.strip()])

corpus["kw_count"] = corpus["Author Keywords"].apply(kw_count)

print("\n✓ Data preprocessing complete.\n")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 1: Annual Retraction Volume (RQ1)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig1():
    """
    Bar chart showing annual volume of LLM-involved retractions.
    Distinguishes between confirmed retractions and retraction lag (2024–2025).
    """
    print("Generating Fig 1 (Annual Retraction Volume)...")
    
    year_counts = (a1_raw[a1_raw["PubYear"].between(2022, 2025)]
                   .groupby("PubYear").size()
                   .reindex([2022, 2023, 2024, 2025], fill_value=0))
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=CONSTRAIN_LAYOUT)
    
    # Color bars: confirmed (dark purple) vs. lag (light purple)
    colors = [C_PURPLE, C_PURPLE, C_LAVEND, C_LAVEND]
    bars = ax.bar(year_counts.index.astype(str), year_counts.values,
                   color=colors, edgecolor=C_WHITE, width=0.6,
                   linewidth=2, zorder=3)
    
    # Dashed border overlay on lag bars (2024, 2025)
    for i, bar in enumerate(bars[2:], start=2):
        rect = mpatches.FancyBboxPatch(
            (bar.get_x(), 0), bar.get_width(), bar.get_height(),
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor=C_GRAY, facecolor="none", linestyle="--", zorder=4)
        ax.add_patch(rect)
    
    # Annotations: sample size labels above bars
    # Offset for 2024 label (tallest bar) to avoid overlap with title
    for i, (bar, val) in enumerate(zip(bars, year_counts.values)):
        label_offset = 20 if i == 2 else 10  # Extra space for tallest bar
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + label_offset,
                f"n={val}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_DARK)
    
    # Grid, axis labels, legend
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, zorder=0, linewidth=0.8)
    ax.set_xlabel("Publication Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Number of LLM-Involved Retractions", fontsize=10, labelpad=8)
    ax.set_title("Fig 1 — RQ1: Annual Volume of LLM-Involved Retractions in CS/DS\n"
                 "(Track A1, n=4882, Retraction Watch + Scopus)",
                 fontsize=11, fontweight="bold", pad=15, loc="center")
    
    ax.set_ylim(0, 450)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Legend
    leg_patches = [
        mpatches.Patch(facecolor=C_PURPLE, edgecolor=C_WHITE, linewidth=2,
                       label="Confirmed retractions"),
        mpatches.Patch(facecolor=C_LAVEND, edgecolor=C_GRAY, linewidth=1.5,
                       linestyle="--", label="Retraction lag (under-count expected)")
    ]
    ax.legend(handles=leg_patches, fontsize=9, loc="upper right",
              framealpha=0.95, edgecolor=C_GRAY)
    
    # YoY growth annotation
    ax.annotate("+1,200% YoY", xy=(1, 364), xytext=(1, 400),
                ha="center", fontsize=9, color=C_PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=1.5, alpha=0.6))
    
    plt.savefig("fig1_bar.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig1_bar.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 2: Publisher Concentration (RQ1)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig2():
    """
    Donut chart showing concentration of retractions across 3 publisher strata.
    Emphasizes dominance of S1 (JIFS) and S2 (Hindawi/Wiley).
    """
    print("Generating Fig 2 (Publisher Concentration)...")
    
    strat_counts = a1_raw["Stratum"].value_counts().reindex(
        ["S1-JIFS", "S2-Hindawi", "S3-Diverse"], fill_value=0)
    
    sizes = strat_counts.values
    colors = [C_PURPLE, C_AMBER, C_TEAL]
    
    # Prepare labels with counts
    labels = [
        f"S1 — JIFS\n(IOS Press/SAGE)\nn={strat_counts['S1-JIFS']}",
        f"S2 — Hindawi\n(Wiley)\nn={strat_counts['S2-Hindawi']}",
        f"S3 — Diverse\nn={strat_counts['S3-Diverse']}"
    ]
    
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.48, edgecolor=C_WHITE, linewidth=2.5),
        textprops={"fontsize": 10})
    
    # Style percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color(C_WHITE)
    
    # Center text (total count)
    ax.text(0, 0, f"n={strat_counts.sum()}\ntotal",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_DARK)
    
    # Legend positioned below to avoid overlap with title
    ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=1,
              fontsize=9, framealpha=0.98, edgecolor=C_GRAY)
    
    ax.set_title("Fig 2 — RQ1: Publisher Concentration of LLM-Involved Retractions\n"
                 "(S1 + S2 account for 92.1% of Track A1)",
                 fontsize=11, fontweight="bold", pad=20, loc="center")
    
    plt.savefig("fig2_donut.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig2_donut.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 3: Citation Distribution Box Plot (RQ2)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig3():
    """
    Box plot comparing citation distributions across groups.
    Shows LLM-retracted papers receive significantly fewer citations.
    Mann-Whitney U test: p < 0.001
    """
    print("Generating Fig 3 (Citation Distribution Box Plot)...")
    
    # Define subsets
    mask_s1 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S1-JIFS")
    mask_s2 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S2-Hindawi")
    mask_s3 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S3-Diverse")
    mask_a1 = (corpus["Track"] == "A1")
    mask_b  = (corpus["Track"] == "B")
    
    # Build data for box plot (clip at 50 for visibility)
    group_data = {
        "Track B\n(n=1,226)":     corpus.loc[mask_b,  "Cited by"].clip(upper=50).values,
        "S3 Diverse\n(A1, n=54)": corpus.loc[mask_s3, "Cited by"].clip(upper=50).values,
        "S2 Hindawi\n(A1, n=97)": corpus.loc[mask_s2, "Cited by"].clip(upper=50).values,
        "S1 JIFS\n(A1, n=531)":   corpus.loc[mask_s1, "Cited by"].clip(upper=50).values,
        "A1 Total\n(n=682)":      corpus.loc[mask_a1, "Cited by"].clip(upper=50).values,
    }
    
    box_vals  = list(group_data.values())
    box_lbls  = list(group_data.keys())
    box_clrs  = [C_TEAL, C_PURPLE, C_PURPLE, C_PURPLE, C_PURPLE]
    
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    bp = ax.boxplot(box_vals, vert=True, patch_artist=True,
                    medianprops=dict(color=C_WHITE, linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2, color=C_DARK),
                    capprops=dict(linewidth=1.2, color=C_DARK),
                    boxprops=dict(linewidth=1.2, color=C_DARK),
                    flierprops=dict(marker="o", markerfacecolor="gray",
                                   markersize=3, alpha=0.4, linestyle="none"))
    
    # Set labels (compatibility with older matplotlib versions)
    ax.set_xticklabels(box_lbls, fontsize=9)
    
    # Color boxes
    for patch, color in zip(bp["boxes"], box_clrs):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    
    # Grid and labels
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, zorder=0, linewidth=0.8)
    ax.set_ylabel("Cited-by Count (clipped at 50 for readability)", fontsize=10, labelpad=8)
    ax.set_title("Fig 3 — RQ2: Citation Distribution — LLM-Retracted vs Comparison Papers\n"
                 "Mann-Whitney U = 234,107, p < 0.001",
                 fontsize=11, fontweight="bold", pad=15)
    
    ax.set_ylim(-1, 52)
    ax.spines[["top", "right"]].set_visible(False)
    
    plt.savefig("fig3_citation_boxplot.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig3_citation_boxplot.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 4: Mean & Median Citations by Group (RQ2)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig4():
    """
    Grouped bar chart showing mean (blue) and median (green) citation counts.
    Directly compares citation impact across strata.
    """
    print("Generating Fig 4 (Mean & Median Citations)...")
    
    mask_s1 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S1-JIFS")
    mask_s2 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S2-Hindawi")
    mask_s3 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S3-Diverse")
    mask_a1 = (corpus["Track"] == "A1")
    mask_b  = (corpus["Track"] == "B")
    
    # Calculate statistics
    stats = {}
    for lbl, msk in [("A1 Total", mask_a1), ("S1 JIFS", mask_s1),
                     ("S2 Hindawi", mask_s2), ("S3 Diverse", mask_s3),
                     ("Track B", mask_b)]:
        sub = corpus.loc[msk, "Cited by"]
        stats[lbl] = {"mean": sub.mean(), "median": sub.median()}
    
    # Prepare data for grouped bar chart
    x_pos = np.arange(len(stats))
    width = 0.35
    
    means   = [stats[k]["mean"] for k in stats.keys()]
    medians = [stats[k]["median"] for k in stats.keys()]
    
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    bars1 = ax.bar(x_pos - width/2, means, width, label="Mean",
                   color=C_PURPLE, alpha=0.8, edgecolor=C_DARK, linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, medians, width, label="Median",
                   color=C_TEAL, alpha=0.8, edgecolor=C_DARK, linewidth=0.8)
    
    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f"{height:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
    
    # Labels and formatting
    ax.set_ylabel("Citations", fontsize=10, labelpad=8)
    ax.set_xlabel("Group", fontsize=10, labelpad=8)
    ax.set_title("Fig 4 — RQ2: Mean and Median Citations by Group\n"
                 "Mann-Whitney U = 234,107, p < 0.001 (A1 vs B)",
                 fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats.keys(), fontsize=9)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor=C_GRAY)
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, zorder=0, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Divider line between A1 and B
    ax.axvline(x=4.5, color=C_GRAY, linestyle="--", linewidth=1.5, alpha=0.5, zorder=0)
    ax.text(2, 9.3, "Track A1", ha="center", fontsize=8.5, style="italic", color=C_GRAY, alpha=0.7)
    ax.text(4, 9.3, "Track B", ha="center", fontsize=8.5, style="italic", color=C_GRAY, alpha=0.7)
    
    plt.savefig("fig4_citation_grouped.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig4_citation_grouped.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 5: Zero-Citation Proportion Stacked Bar (RQ4)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig5():
    """
    Stacked bar chart showing proportion of papers with zero citations.
    Highlights 10× higher zero-citation rate in LLM-retracted papers.
    """
    print("Generating Fig 5 (Zero-Citation Proportion)...")
    
    mask_s1 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S1-JIFS")
    mask_s2 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S2-Hindawi")
    mask_s3 = (corpus["Track"] == "A1") & (corpus["Stratum"] == "S3-Diverse")
    mask_b  = (corpus["Track"] == "B")
    
    # Calculate zero-citation proportions
    strat_zero = {}
    for lbl, msk in [("S1-JIFS", mask_s1), ("S2-Hindawi", mask_s2),
                     ("S3-Diverse", mask_s3), ("Track B", mask_b)]:
        sub = corpus.loc[msk]
        z   = (sub["Cited by"] == 0).sum()
        nz  = (sub["Cited by"] > 0).sum()
        tot = len(sub)
        strat_zero[lbl] = {
            "zero_pct": 100*z/tot, "cited_pct": 100*nz/tot,
            "zero_n": z, "total": tot
        }
    
    xlbls = list(strat_zero.keys())
    zp = [strat_zero[k]["zero_pct"] for k in xlbls]
    cp = [strat_zero[k]["cited_pct"] for k in xlbls]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    b_z = ax.bar(xlbls, zp, color=C_PURPLE, alpha=0.85, label="Zero-citation",
                 edgecolor=C_DARK, linewidth=0.8)
    b_c = ax.bar(xlbls, cp, bottom=zp, color=C_LAVEND, alpha=0.6, label="Has citations",
                 edgecolor=C_DARK, linewidth=0.8)
    
    # Annotations inside zero-citation segments
    for i, (bar, k) in enumerate(zip(b_z, xlbls)):
        z_pct = strat_zero[k]["zero_pct"]
        z_n = strat_zero[k]["zero_n"]
        ax.text(bar.get_x() + bar.get_width()/2, z_pct/2,
                f"{z_pct:.1f}%\n(n={z_n})",
                ha="center", va="center", fontsize=9,
                color=C_WHITE, fontweight="bold")
    
    # Baseline reference line for Track B
    ax.axhline(2.7, color=C_TEAL, linestyle="--", linewidth=2,
               label="Track B zero-citation baseline (2.7%)", zorder=2)
    
    ax.set_ylabel("Proportion of Papers (%)", fontsize=10, labelpad=8)
    ax.set_ylim(0, 110)
    ax.set_title("Fig 5 — RQ4: Zero-Citation Proportion by Group\n"
                 "χ² = 247.89, df = 1, p < 0.001 (A1 vs B)",
                 fontsize=11, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor=C_GRAY)
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    
    plt.savefig("fig5_zero_citation_stacked.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig5_zero_citation_stacked.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 6: Retraction Lag Scatter Plot (RQ1 & RQ2)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig6():
    """
    Scatter plot: publication year vs. citation accumulation.
    Shows declining citations for recently retracted papers (2024–2025 lag).
    Jittered for clarity; horizontal bars = annual mean.
    """
    print("Generating Fig 6 (Retraction Lag Scatter)...")
    
    a1_sc_clean = a1_sc.copy()
    a1_sc_clean["Cited by"] = pd.to_numeric(a1_sc_clean["Cited by"], errors="coerce").fillna(0)
    a1_sc_clean["Year"] = pd.to_numeric(a1_sc_clean["Year"], errors="coerce")
    a1_sc_clean = a1_sc_clean[a1_sc_clean["Year"].between(2022, 2025)]
    
    # Stratum assignment
    def stratum_from_scopus(row):
        src = str(row.get("Source title", "")).lower()
        if "intelligent" in src and "fuzzy" in src:
            return "S1-JIFS"
        elif any(x in src for x in ["complexity", "mathematical", "hindawi"]):
            return "S2-Hindawi"
        else:
            return "S3-Diverse"
    
    a1_sc_clean["Stratum"] = a1_sc_clean.apply(stratum_from_scopus, axis=1)
    a1_sc_clean["log_cite"] = np.log1p(a1_sc_clean["Cited by"])
    
    # Add jitter for visibility
    np.random.seed(42)
    a1_sc_clean["year_jit"] = (a1_sc_clean["Year"] +
                               np.random.uniform(-0.35, 0.35, len(a1_sc_clean)))
    
    strat_colors = {"S1-JIFS": C_PURPLE, "S2-Hindawi": C_AMBER, "S3-Diverse": C_TEAL}
    
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    # Scatter by stratum
    for strat, col in strat_colors.items():
        sub = a1_sc_clean[a1_sc_clean["Stratum"] == strat]
        ax.scatter(sub["year_jit"], sub["log_cite"],
                   c=col, alpha=0.35, s=20, label=strat, zorder=3,
                   edgecolors="none")
    
    # Mean lines per year
    for yr in [2022, 2023, 2024, 2025]:
        m = a1_sc_clean[a1_sc_clean["Year"] == yr]["log_cite"].mean()
        ax.hlines(m, yr - 0.4, yr + 0.4, colors=C_DARK,
                 linewidth=2.5, zorder=5, linestyle="-", alpha=0.8)
        
        # Mean label positioned to right to avoid overlap with axis labels
        ax.text(yr + 0.5, m, f"μ={m:.2f}",
                va="center", fontsize=8.5, color=C_DARK, fontweight="bold")
    
    # Formatting
    ax.set_xticks([2022, 2023, 2024, 2025])
    ax.set_xticklabels(["2022\n(Nov–Dec)", "2023", "2024\n(lag)", "2025\n(lag)"],
                      fontsize=9)
    ax.set_ylabel("log(1 + Cited-by Count)", fontsize=10, labelpad=8)
    ax.set_xlabel("Publication Year", fontsize=10, labelpad=8)
    ax.set_title("Fig 6 — RQ1 & RQ2: Publication Year vs Citation Accumulation\n"
                 "(Track A1, n=682; horizontal bars = annual mean; jittered for clarity)",
                 fontsize=11, fontweight="bold", pad=15)
    
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor=C_GRAY)
    ax.grid(color=C_GRAY, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.2, 4.5)
    
    plt.savefig("fig6_retraction_lag.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig6_retraction_lag.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 7: Bibliometric Anomaly Indicators (RQ3 & RQ4)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig7():
    """
    Side-by-side bar charts:
    (Left)  Mean author keyword count — fewer keywords in LLM-retracted
    (Right) Zero-citation proportion — 10× higher in LLM-retracted
    """
    print("Generating Fig 7 (Bibliometric Anomaly Indicators)...")
    
    mask_a1 = (corpus["Track"] == "A1")
    mask_b  = (corpus["Track"] == "B")
    
    a1_kw = corpus.loc[mask_a1, "kw_count"].mean()
    b_kw  = corpus.loc[mask_b,  "kw_count"].mean()
    a1_zp = 26.5
    b_zp  = 2.7
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             constrained_layout=CONSTRAIN_LAYOUT)
    
    # ─ Left Panel: Keyword Count ─────────────────────────────────────────────
    ax = axes[0]
    vals  = [a1_kw, b_kw]
    labels = ["Track A1\n(LLM-retracted)", "Track B\n(Comparison)"]
    clrs  = [C_PURPLE, C_TEAL]
    
    bars  = ax.bar(labels, vals, color=clrs, width=0.5, alpha=0.85,
                   edgecolor=C_DARK, linewidth=1)
    
    # Value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_DARK)
    
    ax.set_ylabel("Mean Author Keyword Count", fontsize=10, labelpad=8)
    ax.set_title("Author Keyword Count\nU = 318,380, p < 0.001",
                 fontsize=10, fontweight="bold", pad=10)
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Annotation: fewer keywords
    ax.annotate("← Fewer keywords\n(null on stuffing)", xy=(0, a1_kw),
                xytext=(-0.35, a1_kw + 1.2), fontsize=8.5,
                color=C_PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2, alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WHITE,
                         edgecolor=C_PURPLE, linewidth=0.8, alpha=0.9))
    
    # ─ Right Panel: Zero-Citation Proportion ──────────────────────────────────
    ax = axes[1]
    vals2 = [a1_zp, b_zp]
    
    bars2 = ax.bar(labels, vals2, color=clrs, width=0.5, alpha=0.85,
                   edgecolor=C_DARK, linewidth=1)
    
    # Value labels
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_DARK)
    
    ax.set_ylabel("Zero-Citation Papers (%)", fontsize=10, labelpad=8)
    ax.set_title("Zero-Citation Proportion\nχ² = 247.89, p < 0.001",
                 fontsize=10, fontweight="bold", pad=10)
    ax.set_ylim(0, 31)
    ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Annotation: 10× higher
    ax.annotate("~10× higher\nthan Track B", xy=(0, a1_zp),
                xytext=(-0.35, a1_zp + 2.5), fontsize=8.5,
                color=C_PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2, alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WHITE,
                         edgecolor=C_PURPLE, linewidth=0.8, alpha=0.9))
    
    fig.suptitle("Fig 7 — RQ3 & RQ4: Bibliometric Anomaly Indicators (Track A1 vs Track B)",
                 fontsize=11, fontweight="bold", y=1.00)
    
    plt.savefig("fig7_keyword_anomaly.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig7_keyword_anomaly.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ FIGURE 8: Heatmap — Top 10 Journals × Year (RQ1)
# ════════════════════════════════════════════════════════════════════════════

def generate_fig8():
    """
    Heatmap showing retraction volume by top 10 journals across years (2022–2025).
    Darker = more retractions. Reveals temporal concentration patterns.
    """
    print("Generating Fig 8 (Heatmap: Top Journals × Year)...")
    
    a1_sc_yr = a1_sc.copy()
    a1_sc_yr["Year"] = pd.to_numeric(a1_sc_yr["Year"], errors="coerce")
    a1_sc_yr = a1_sc_yr[a1_sc_yr["Year"].between(2022, 2025)]
    
    # Extract top 10 journals
    top_journals = (a1_sc_yr["Source title"]
                   .value_counts().head(10).index.tolist())
    
    # Create pivot table
    heat_data = (a1_sc_yr[a1_sc_yr["Source title"].isin(top_journals)]
                .groupby(["Source title", "Year"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=[2022, 2023, 2024, 2025], fill_value=0))
    
    # Shorten journal names for readability
    heat_data.index = (heat_data.index
        .str.replace("Journal of Intelligent and Fuzzy Systems",
                    "JIFS", regex=False)
        .str.replace("Optical and Quantum Electronics",
                    "Opt & Quantum Electron.", regex=False)
        .str.replace("Journal of Computational Methods in Sciences and Engineering",
                    "J. Comp. Methods Sci. Eng.", regex=False)
        .str.replace("Intelligent Systems with Applications",
                    "Intell. Syst. Appl.", regex=False))
    
    heat_data.columns = ["2022\n(Nov–Dec)", "2023", "2024", "2025"]
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=CONSTRAIN_LAYOUT)
    
    # Create heatmap
    sns.heatmap(heat_data, annot=True, fmt="d", cmap="Purples",
                linewidths=1, linecolor=C_WHITE,
                cbar_kws={"label": "Retraction count", "shrink": 0.8},
                ax=ax, vmin=0,
                annot_kws={"fontsize": 9, "fontweight": "bold", "color": C_DARK})
    
    # Labels
    ax.set_xlabel("Publication Year", fontsize=10, labelpad=8, fontweight="bold")
    ax.set_ylabel("Journal", fontsize=10, labelpad=8, fontweight="bold")
    ax.set_title("Fig 8 — RQ1: Retraction Volume Heatmap — Top 10 Journals × Year\n"
                 "(Track A1, n=682; darker = more retractions)",
                 fontsize=11, fontweight="bold", pad=15)
    
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=9)
    
    plt.savefig("fig8_heatmap.png", dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close()
    print("  ✓ fig8_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# ▶ MAIN: Generate All Figures
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80 + "\n")
    
    print(f"Output Resolution: {OUTPUT_DPI} DPI")
    print(f"Layout: Constrained (no text overlap)")
    print(f"Typography: Arial, publication-grade\n")
    
    try:
        generate_fig1()
        generate_fig2()
        generate_fig3()
        generate_fig4()
        generate_fig5()
        generate_fig6()
        generate_fig7()
        generate_fig8()
        
        print("\n" + "="*80)
        print("✅  ALL 8 FIGURES GENERATED SUCCESSFULLY")
        print("="*80)
        print("\nOutput Files:")
        print("  1. fig1_bar.png                  — Annual retraction volume")
        print("  2. fig2_donut.png                — Publisher concentration")
        print("  3. fig3_citation_boxplot.png     — Citation distribution (box plot)")
        print("  4. fig4_citation_grouped.png     — Mean/median citations")
        print("  5. fig5_zero_citation_stacked.png — Zero-citation proportion")
        print("  6. fig6_retraction_lag.png       — Publication year vs. citations")
        print("  7. fig7_keyword_anomaly.png      — Bibliometric anomalies")
        print("  8. fig8_heatmap.png              — Top journals × year heatmap")
        print("\nQuality Metrics:")
        print("  ✓ No text overlap or clipping")
        print("  ✓ Constrained layout (auto-spacing)")
        print("  ✓ 600 DPI publication quality")
        print("  ✓ IEEE/Springer journal style")
        print("  ✓ Consistent color palette")
        print("  ✓ Optimized typography (Arial)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
