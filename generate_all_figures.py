"""
generate_all_figures.py
Run: python generate_all_figures.py
Outputs: fig1_bar.png, fig2_donut.png, fig3_citation_boxplot.png,
         fig4_citation_grouped.png, fig5_zero_citation_stacked.png,
         fig6_retraction_lag.png, fig7_keyword_anomaly.png, fig8_heatmap.png
Requires: pandas, matplotlib, seaborn, numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Colours (consistent palette) ────────────────────────────────────────────
C_PURPLE  = "#534AB7"   # Track A1 / primary
C_TEAL    = "#1D9E75"   # Track B / comparison
C_AMBER   = "#EF9F27"   # S2 Hindawi
C_LAVEND  = "#C5C2ED"   # light purple accent
C_GRAY    = "#BFBFBF"

# ── Load data ────────────────────────────────────────────────────────────────
a1_raw  = pd.read_csv("A1_final.csv")
a1_sc   = pd.read_csv("A1_scopus_merged.csv")
corpus  = pd.read_csv("analysis_corpus.csv")

# ── Derive Stratum from Publisher string ─────────────────────────────────────
def assign_stratum(pub):
    pub = str(pub).lower()
    if "ios press" in pub or "sage" in pub:
        return "S1-JIFS"
    elif "hindawi" in pub or "wiley" in pub:
        return "S2-Hindawi"
    else:
        return "S3-Diverse"

a1_raw["Stratum"] = a1_raw["Publisher"].apply(assign_stratum)

# ── Parse publication year from A1_final ────────────────────────────────────
a1_raw["PubDate"]  = pd.to_datetime(a1_raw["OriginalPaperDate"], errors="coerce")
a1_raw["PubYear"]  = a1_raw["PubDate"].dt.year

# ── Derive Track & keyword count for analysis_corpus ────────────────────────
# Track A1 papers have "RETRACTION" or "RETRACTED" in their title
corpus["Track"] = corpus["Title"].str.contains(
    r"RETRACT", case=False, na=False).map({True: "A1", False: "B"})

# Stratum for A1 rows in corpus
def stratum_from_source(row):
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

# Keyword count
def kw_count(kw_str):
    if pd.isna(kw_str) or str(kw_str).strip() == "":
        return 0
    return len([k for k in str(kw_str).split(";") if k.strip()])

corpus["kw_count"] = corpus["Author Keywords"].apply(kw_count)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Annual retraction volume (Track A1 only)
# ═══════════════════════════════════════════════════════════════════════════
year_counts = (a1_raw[a1_raw["PubYear"].between(2022, 2025)]
               .groupby("PubYear").size().reindex([2022,2023,2024,2025], fill_value=0))

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(year_counts.index.astype(str), year_counts.values,
              color=[C_PURPLE, C_PURPLE, C_LAVEND, C_LAVEND],
              edgecolor="white", width=0.55, zorder=3)

for bar, val in zip(bars, year_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"n={val}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# dashed border on lag bars
for i, bar in enumerate(bars[2:], start=2):
    rect = mpatches.FancyBboxPatch(
        (bar.get_x(), 0), bar.get_width(), bar.get_height(),
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor=C_GRAY, facecolor="none", linestyle="--", zorder=4)
    ax.add_patch(rect)

ax.set_xlabel("Publication Year", fontsize=12)
ax.set_ylabel("Number of LLM-Involved Retractions", fontsize=12)
ax.set_title("Fig 1 — RQ1: Annual Volume of LLM-Involved Retractions in CS/DS\n"
             "(Track A1, n = 682, Retraction Watch + Scopus)", fontsize=11, pad=10)
ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.5, zorder=0)
ax.set_ylim(0, 430)
ax.spines[["top","right"]].set_visible(False)

leg = [mpatches.Patch(facecolor=C_PURPLE, label="Confirmed retractions"),
       mpatches.Patch(facecolor=C_LAVEND, edgecolor=C_GRAY, linestyle="--",
                      linewidth=1.5, label="Retraction lag (under-count expected)")]
ax.legend(handles=leg, fontsize=9, loc="upper right")
ax.annotate("+1,200% YoY", xy=("2023", 364), xytext=("2023", 390),
            ha="center", fontsize=9, color=C_PURPLE, fontweight="bold")

plt.tight_layout()
plt.savefig("fig1_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig1_bar.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Publisher concentration donut
# ═══════════════════════════════════════════════════════════════════════════
strat_counts = a1_raw["Stratum"].value_counts().reindex(
    ["S1-JIFS","S2-Hindawi","S3-Diverse"], fill_value=0)

labels = [f"S1 — JIFS\n(IOS Press/SAGE)\nn={strat_counts['S1-JIFS']}",
          f"S2 — Hindawi\n(Wiley)\nn={strat_counts['S2-Hindawi']}",
          f"S3 — Diverse\nn={strat_counts['S3-Diverse']}"]
sizes  = strat_counts.values
colors = [C_PURPLE, C_AMBER, C_TEAL]

fig, ax = plt.subplots(figsize=(6, 5.5))
wedges, texts, autotexts = ax.pie(
    sizes, labels=None, colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2))

for at in autotexts:
    at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")

ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5,-0.12),
          ncol=1, fontsize=9, framealpha=0)
ax.text(0, 0, f"n={strat_counts.sum()}\ntotal", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#333333")
ax.set_title("Fig 2 — RQ1: Publisher Concentration of LLM-Involved Retractions\n"
             "(S1 + S2 account for 92.1% of Track A1)", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("fig2_donut.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig2_donut.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Citation distribution box plot (NEW — replaces plain grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
# Build subset for box plot
box_data = []
groups   = []

# A1 by stratum + B overall
mask_s1 = (corpus["Track"]=="A1") & (corpus["Stratum"]=="S1-JIFS")
mask_s2 = (corpus["Track"]=="A1") & (corpus["Stratum"]=="S2-Hindawi")
mask_s3 = (corpus["Track"]=="A1") & (corpus["Stratum"]=="S3-Diverse")
mask_a1 = (corpus["Track"]=="A1")
mask_b  = (corpus["Track"]=="B")

group_map = {
    "Track B\n(n=1,226)":      corpus.loc[mask_b,  "Cited by"].clip(upper=50).values,
    "S3 Diverse\n(A1, n=54)":  corpus.loc[mask_s3, "Cited by"].clip(upper=50).values,
    "S2 Hindawi\n(A1, n=97)":  corpus.loc[mask_s2, "Cited by"].clip(upper=50).values,
    "S1 JIFS\n(A1, n=531)":    corpus.loc[mask_s1, "Cited by"].clip(upper=50).values,
    "A1 Total\n(n=682)":       corpus.loc[mask_a1, "Cited by"].clip(upper=50).values,
}

box_vals  = list(group_map.values())
box_lbls  = list(group_map.keys())
box_clrs  = [C_TEAL, C_PURPLE, C_PURPLE, C_PURPLE, C_PURPLE]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_vals, vert=True, patch_artist=True,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=2.5,
                                markerfacecolor=C_GRAY, alpha=0.4))

for patch, col in zip(bp["boxes"], box_clrs):
    patch.set_facecolor(col); patch.set_alpha(0.85)

ax.set_xticklabels(box_lbls, fontsize=9)
ax.set_ylabel("Cited-by Count (clipped at 50 for readability)", fontsize=10)
ax.set_title("Fig 3 — RQ2: Citation Distribution — LLM-Retracted vs Comparison Papers\n"
             "Mann-Whitney U = 234,107, p < 0.001", fontsize=11, pad=10)
ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.5)
ax.spines[["top","right"]].set_visible(False)

leg = [mpatches.Patch(facecolor=C_PURPLE, label="Track A1 (LLM-retracted)"),
       mpatches.Patch(facecolor=C_TEAL,   label="Track B (comparison)")]
ax.legend(handles=leg, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("fig3_citation_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig3_citation_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Citation grouped bar (mean + median) — keep original chart upgraded
# ═══════════════════════════════════════════════════════════════════════════
grp_labels = ["A1 Total", "S1 JIFS", "S2 Hindawi", "S3 Diverse", "Track B"]
means   = [4.35, 3.70, 5.82, 8.02, 7.00]
medians = [2.00, 2.00, 2.00, 4.00, 4.00]
x = np.arange(len(grp_labels)); w = 0.36

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, means,   w, label="Mean",   color=C_PURPLE, alpha=0.9)
b2 = ax.bar(x + w/2, medians, w, label="Median", color=C_TEAL,   alpha=0.9)

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8.5)

ax.axvline(3.5, color=C_GRAY, linestyle="--", linewidth=1, label="A1 | B divider")
ax.set_xticks(x); ax.set_xticklabels(grp_labels, fontsize=10)
ax.set_ylabel("Citations", fontsize=11)
ax.set_title("Fig 4 — RQ2: Mean and Median Citations by Group\n"
             "Mann-Whitney U = 234,107, p < 0.001 (A1 vs B)", fontsize=11, pad=10)
ax.legend(fontsize=9); ax.set_ylim(0, 11)
ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.5)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig4_citation_grouped.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig4_citation_grouped.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Zero-citation stacked bar by stratum (NEW)
# ═══════════════════════════════════════════════════════════════════════════
strat_zero = {}
for lbl, mask in [("S1-JIFS", mask_s1), ("S2-Hindawi", mask_s2),
                  ("S3-Diverse", mask_s3), ("Track B", mask_b)]:
    sub  = corpus.loc[mask]
    z    = (sub["Cited by"] == 0).sum()
    nz   = (sub["Cited by"] >  0).sum()
    tot  = len(sub)
    strat_zero[lbl] = {"zero_pct": 100*z/tot, "cited_pct": 100*nz/tot,
                       "zero_n": z, "total": tot}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
xlbls = list(strat_zero.keys())
zp    = [strat_zero[k]["zero_pct"]  for k in xlbls]
cp    = [strat_zero[k]["cited_pct"] for k in xlbls]

b_z = ax.bar(xlbls, zp, color=C_PURPLE, alpha=0.9, label="Zero-citation")
b_c = ax.bar(xlbls, cp, bottom=zp, color=C_LAVEND, alpha=0.7, label="Has citations")

for i, (bar, k) in enumerate(zip(b_z, xlbls)):
    ax.text(bar.get_x()+bar.get_width()/2,
            strat_zero[k]["zero_pct"]/2,
            f"{strat_zero[k]['zero_pct']:.1f}%\n(n={strat_zero[k]['zero_n']})",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.axhline(2.7, color=C_TEAL, linestyle="--", linewidth=1.5,
           label="Track B zero-citation baseline (2.7%)")
ax.set_ylabel("Proportion of Papers (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.set_title("Fig 5 — RQ4: Zero-Citation Proportion by Group\n"
             "χ² = 247.89, df = 1, p < 0.001 (A1 vs B)", fontsize=11, pad=10)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig5_zero_citation_stacked.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig5_zero_citation_stacked.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — Retraction lag scatter (pub year vs citation count) (NEW)
# ═══════════════════════════════════════════════════════════════════════════
a1_sc_clean = a1_sc.copy()
a1_sc_clean["Cited by"] = pd.to_numeric(a1_sc_clean["Cited by"], errors="coerce").fillna(0)
a1_sc_clean["Year"] = pd.to_numeric(a1_sc_clean["Year"], errors="coerce")
a1_sc_clean = a1_sc_clean[a1_sc_clean["Year"].between(2022, 2025)]

# Assign stratum
def stratum_from_scopus(row):
    src = str(row.get("Source title","")).lower()
    if "intelligent" in src and "fuzzy" in src:
        return "S1-JIFS"
    elif any(x in src for x in ["complexity","mathematical problem","abstract","hindawi"]):
        return "S2-Hindawi"
    else:
        return "S3-Diverse"

a1_sc_clean["Stratum"] = a1_sc_clean.apply(stratum_from_scopus, axis=1)
a1_sc_clean["log_cite"] = np.log1p(a1_sc_clean["Cited by"])

# Jitter year
np.random.seed(42)
a1_sc_clean["year_jit"] = a1_sc_clean["Year"] + np.random.uniform(-0.35, 0.35,
                                                                    len(a1_sc_clean))

strat_colors = {"S1-JIFS": C_PURPLE, "S2-Hindawi": C_AMBER, "S3-Diverse": C_TEAL}

fig, ax = plt.subplots(figsize=(8.5, 5))
for strat, col in strat_colors.items():
    sub = a1_sc_clean[a1_sc_clean["Stratum"] == strat]
    ax.scatter(sub["year_jit"], sub["log_cite"],
               c=col, alpha=0.35, s=18, label=strat, zorder=3)

# Overlay mean per year
for yr in [2022, 2023, 2024, 2025]:
    m = a1_sc_clean[a1_sc_clean["Year"]==yr]["log_cite"].mean()
    ax.hlines(m, yr-0.4, yr+0.4, colors="#333333", linewidth=2.5, zorder=5)
    ax.text(yr+0.42, m, f"μ={m:.2f}", va="center", fontsize=8.5, color="#333333")

ax.set_xticks([2022,2023,2024,2025])
ax.set_xticklabels(["2022\n(Nov–Dec)", "2023", "2024\n(lag)", "2025\n(lag)"], fontsize=10)
ax.set_ylabel("log(1 + Cited-by Count)", fontsize=11)
ax.set_xlabel("Publication Year", fontsize=11)
ax.set_title("Fig 6 — RQ1 & RQ2: Publication Year vs Citation Accumulation\n"
             "(Track A1, n=682; horizontal bars = annual mean; jittered for clarity)",
             fontsize=11, pad=10)
ax.legend(fontsize=9, loc="upper right")
ax.grid(color=C_GRAY, linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig6_retraction_lag.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig6_retraction_lag.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 — Bibliometric anomaly indicators (keyword count + zero-citation %)
# ═══════════════════════════════════════════════════════════════════════════
a1_kw = corpus.loc[mask_a1, "kw_count"].mean()
b_kw  = corpus.loc[mask_b,  "kw_count"].mean()
a1_zp = 26.5
b_zp  = 2.7

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

# Left: keyword count
ax = axes[0]
vals  = [a1_kw, b_kw]
clrs  = [C_PURPLE, C_TEAL]
bars  = ax.bar(["Track A1\n(LLM-retracted)", "Track B\n(Comparison)"],
               vals, color=clrs, width=0.45, alpha=0.9)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Author Keyword Count", fontsize=10)
ax.set_title("Author Keyword Count\nU = 318,380, p < 0.001", fontsize=10, pad=8)
ax.set_ylim(0, 7); ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.5)
ax.spines[["top","right"]].set_visible(False)
ax.annotate("← Fewer keywords\n   (null on stuffing)", xy=(0, a1_kw),
            xytext=(0.15, a1_kw+1.2), fontsize=8, color=C_PURPLE,
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1))

# Right: zero-citation proportion
ax = axes[1]
vals2 = [a1_zp, b_zp]
bars2 = ax.bar(["Track A1\n(LLM-retracted)", "Track B\n(Comparison)"],
               vals2, color=clrs, width=0.45, alpha=0.9)
for bar, v in zip(bars2, vals2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Zero-Citation Papers (%)", fontsize=10)
ax.set_title("Zero-Citation Proportion\nχ² = 247.89, p < 0.001", fontsize=10, pad=8)
ax.set_ylim(0, 38); ax.grid(axis="y", color=C_GRAY, linestyle="--", alpha=0.5)
ax.spines[["top","right"]].set_visible(False)
ax.annotate("~10× higher\n   than Track B", xy=(0, a1_zp),
            xytext=(0.15, a1_zp+3), fontsize=8, color=C_PURPLE,
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1))

fig.suptitle("Fig 7 — RQ3 & RQ4: Bibliometric Anomaly Indicators (Track A1 vs Track B)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("fig7_keyword_anomaly.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig7_keyword_anomaly.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 — Heatmap: top journals × year × retraction count (NEW)
# ═══════════════════════════════════════════════════════════════════════════
a1_sc_yr = a1_sc_clean.copy()
top_journals = (a1_sc_yr["Source title"]
                .value_counts().head(10).index.tolist())

heat_data = (a1_sc_yr[a1_sc_yr["Source title"].isin(top_journals)]
             .groupby(["Source title","Year"])
             .size().unstack(fill_value=0)
             .reindex(columns=[2022,2023,2024,2025], fill_value=0))

# Shorten long journal names
heat_data.index = (heat_data.index
    .str.replace("Journal of Intelligent and Fuzzy Systems", "JIFS", regex=False)
    .str.replace("Optical and Quantum Electronics", "Opt & Quantum Electron.", regex=False)
    .str.replace("Journal of Computational Methods in Sciences and Engineering",
                 "J. Comp. Methods Sci. Eng.", regex=False)
    .str.replace("Intelligent Systems with Applications",
                 "Intell. Syst. Appl.", regex=False))

heat_data.columns = ["2022\n(Nov–Dec)", "2023", "2024", "2025"]

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.heatmap(heat_data, annot=True, fmt="d", cmap="Purples",
            linewidths=0.5, linecolor="#dddddd",
            cbar_kws={"label":"Retraction count", "shrink":0.7},
            ax=ax)
ax.set_xlabel("Publication Year", fontsize=11)
ax.set_ylabel("")
ax.set_title("Fig 8 — RQ1: Retraction Volume Heatmap — Top 10 Journals × Year\n"
             "(Track A1, n=682; darker = more retractions)", fontsize=11, pad=10)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig("fig8_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig8_heatmap.png")


print("\n✅  All 8 figures saved successfully.")
print("Files: fig1_bar.png  fig2_donut.png  fig3_citation_boxplot.png")
print("       fig4_citation_grouped.png  fig5_zero_citation_stacked.png")
print("       fig6_retraction_lag.png  fig7_keyword_anomaly.png  fig8_heatmap.png")