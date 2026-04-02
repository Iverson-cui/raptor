# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.lines import Line2D

# # ==========================================
# # 1. ACADEMIC STYLE CONFIGURATION
# # ==========================================
# plt.rcParams.update(
#     {
#         "font.size": 12,
#         "font.family": "serif",  # Standard for academic papers
#         "axes.labelsize": 14,
#         "axes.titlesize": 14,
#         "legend.fontsize": 10,
#         "xtick.labelsize": 11,
#         "ytick.labelsize": 11,
#     }
# )

# # Colors and Markers
# COLOR_SQUAD = "#D32F2F"  # Academic Red
# COLOR_TQA = "#1976D2"  # Academic Blue
# MARKER_SCHEME1 = "*"  # Star
# MARKER_SCHEME2 = "D"  # Diamond
# MARKER_BASELINE = "o"  # Circle for the shared baseline
# SIZE_MARKER = 8
# SIZE_STAR = 12  # Stars usually need to be a bit larger to match visual weight

# # ==========================================
# # 2. DATA PLACEHOLDERS (Replace with real data)
# # ==========================================

# # SQuAD Data
# squad_x = [64, 128, 256, 512]

# # SQuAD F1 Scores: [Baseline(64), Scheme1(128), Scheme1(256)] / [Baseline(64), Scheme2(128), Scheme2(256)]
# squad_f1_baseline = 47.64
# squad_f1_scheme1 = [squad_f1_baseline, 49.59, 46.85, 37.64]
# squad_f1_scheme2 = [squad_f1_baseline, 44.09, 45.41, 35.65]

# # SQuAD EM Scores
# squad_em_baseline = 30.00
# squad_em_scheme1 = [squad_em_baseline, 32.5, 29, 18.33]
# squad_em_scheme2 = [squad_em_baseline, 27.5, 31, 16.67]

# # TriviaQA Data
# tqa_x = [256, 512, 1024, 2048]

# # TriviaQA F1 Scores: [Baseline(256), Scheme1(512), Scheme1(1024)] / [Baseline(256), Scheme2(512), Scheme2(1024)]
# tqa_f1_baseline = 41.06
# tqa_f1_scheme1 = [tqa_f1_baseline, 38.44, 41.49, 33.78]
# tqa_f1_scheme2 = [tqa_f1_baseline, 39.44, 40.03, 35.0]

# # TriviaQA EM Scores
# tqa_em_baseline = 25.33
# tqa_em_scheme1 = [tqa_em_baseline, 24.0, 27.0, 20.67]
# tqa_em_scheme2 = [tqa_em_baseline, 25, 25.5, 22.0]

# # ==========================================
# # 3. FIGURE SETUP & PLOTTING
# # ==========================================
# fig, ax = plt.subplots(figsize=(9, 6))

# # --- Plot SQuAD (Red) ---
# # F1 (Solid lines)
# ax.plot(
#     squad_x,
#     squad_f1_scheme1,
#     color=COLOR_SQUAD,
#     linestyle="-",
#     marker=MARKER_SCHEME1,
#     markersize=SIZE_STAR,
#     label="SQuAD F1 (Scheme 1)",
# )
# ax.plot(
#     squad_x,
#     squad_f1_scheme2,
#     color=COLOR_SQUAD,
#     linestyle="-",
#     marker=MARKER_SCHEME2,
#     markersize=SIZE_MARKER,
#     label="SQuAD F1 (Scheme 2)",
# )
# # EM (Dashed lines)
# ax.plot(
#     squad_x,
#     squad_em_scheme1,
#     color=COLOR_SQUAD,
#     linestyle="--",
#     marker=MARKER_SCHEME1,
#     markersize=SIZE_STAR,
#     label="SQuAD EM (Scheme 1)",
# )
# ax.plot(
#     squad_x,
#     squad_em_scheme2,
#     color=COLOR_SQUAD,
#     linestyle="--",
#     marker=MARKER_SCHEME2,
#     markersize=SIZE_MARKER,
#     label="SQuAD EM (Scheme 2)",
# )

# # --- Plot TriviaQA (Blue) ---
# # F1 (Solid lines)
# ax.plot(
#     tqa_x,
#     tqa_f1_scheme1,
#     color=COLOR_TQA,
#     linestyle="-",
#     marker=MARKER_SCHEME1,
#     markersize=SIZE_STAR,
#     label="TQA F1 (Scheme 1)",
# )
# ax.plot(
#     tqa_x,
#     tqa_f1_scheme2,
#     color=COLOR_TQA,
#     linestyle="-",
#     marker=MARKER_SCHEME2,
#     markersize=SIZE_MARKER,
#     label="TQA F1 (Scheme 2)",
# )
# # EM (Dashed lines)
# ax.plot(
#     tqa_x,
#     tqa_em_scheme1,
#     color=COLOR_TQA,
#     linestyle="--",
#     marker=MARKER_SCHEME1,
#     markersize=SIZE_STAR,
#     label="TQA EM (Scheme 1)",
# )
# ax.plot(
#     tqa_x,
#     tqa_em_scheme2,
#     color=COLOR_TQA,
#     linestyle="--",
#     marker=MARKER_SCHEME2,
#     markersize=SIZE_MARKER,
#     label="TQA EM (Scheme 2)",
# )

# # --- Plot Baselines clearly ---
# # Plotting the baseline points again on top as a neutral dot to show where lines branch from
# ax.plot(
#     squad_x[0],
#     squad_f1_baseline,
#     color=COLOR_SQUAD,
#     marker=MARKER_BASELINE,
#     markersize=SIZE_MARKER,
#     zorder=5,
# )
# ax.plot(
#     squad_x[0],
#     squad_em_baseline,
#     color=COLOR_SQUAD,
#     marker=MARKER_BASELINE,
#     markersize=SIZE_MARKER,
#     zorder=5,
# )
# ax.plot(
#     tqa_x[0],
#     tqa_f1_baseline,
#     color=COLOR_TQA,
#     marker=MARKER_BASELINE,
#     markersize=SIZE_MARKER,
#     zorder=5,
# )
# ax.plot(
#     tqa_x[0],
#     tqa_em_baseline,
#     color=COLOR_TQA,
#     marker=MARKER_BASELINE,
#     markersize=SIZE_MARKER,
#     zorder=5,
# )

# # ==========================================
# # 4. AXES FORMATTING & STYLING
# # ==========================================
# # Use Log base 2 scale for X-axis to evenly space 64, 128, 256, 512, 1024
# ax.set_xscale("log", base=2)
# xticks = [64, 128, 256, 512, 1024, 2048]
# ax.set_xticks(xticks)
# # Format ticks as standard numbers (not scientific notation)
# ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

# # Labels
# ax.set_xlabel("Chunk Size (Tokens)", fontweight="bold")
# ax.set_ylabel("Score (%)", fontweight="bold")

# # Adjust Y-axis limits depending on your exact data
# ax.set_ylim(10, 70)

# # Grid
# ax.grid(True, which="major", linestyle="-", alpha=0.3)
# ax.grid(True, which="minor", linestyle="--", alpha=0.1)

# # Legend Configuration
# # Placing the legend outside the plot area on the right, matching your sketch
# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1.03, 0.5),
#     frameon=True,
#     edgecolor="black",
#     title="Metrics & Schemes",
# )

# plt.tight_layout()

# # ==========================================
# # 5. EXPORT AND DISPLAY
# # ==========================================
# # Save as PDF for the LaTeX/paper compilation, PNG for quick viewing
# plt.savefig("rag_chunk_performance.pdf", format="pdf", dpi=600, bbox_inches="tight")
# plt.savefig("rag_chunk_performance.png", format="png", dpi=600, bbox_inches="tight")
# plt.savefig("rag_chunk_performance.svg", format="svg", dpi=600, bbox_inches="tight")

# plt.show()


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ==========================================
# 1. ACADEMIC STYLE CONFIGURATION
# ==========================================
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",  # Standard for academic papers
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,  # Requirement 4: Match legend font size
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

# Colors and Markers
COLOR_SQUAD = "#D32F2F"
COLOR_TQA = "#1976D2"
MARKER_SCHEME1 = "*"
MARKER_SCHEME2 = "D"
MARKER_BASELINE = "o"

# ==========================================
# 2. DATA PLACEHOLDERS (Extended to 2048)
# ==========================================

# # SQuAD Data
# squad_x = [64, 128, 256, 512]
# squad_f1_scheme1 = [60.0, 62.0, 58.0, 48.0]
# squad_f1_scheme2 = [60.0, 55.0, 56.0, 45.0]
# squad_em_scheme1 = [45.0, 48.0, 44.0, 32.0]
# squad_em_scheme2 = [45.0, 42.0, 46.0, 30.0]

# # TriviaQA Data
# tqa_x = [256, 512, 1024, 2048]
# tqa_f1_scheme1 = [52.0, 48.0, 54.0, 42.0]
# tqa_f1_scheme2 = [52.0, 50.0, 51.0, 45.0]
# tqa_em_scheme1 = [40.0, 38.0, 42.0, 32.0]
# tqa_em_scheme2 = [40.0, 39.0, 40.0, 35.0]

# SQuAD Data
squad_x = [64, 128, 256, 512]

# SQuAD F1 Scores: [Baseline(64), Scheme1(128), Scheme1(256)] / [Baseline(64), Scheme2(128), Scheme2(256)]
squad_f1_baseline = 47.64
squad_f1_scheme1 = [squad_f1_baseline, 49.59, 46.85, 37.64]
squad_f1_scheme2 = [squad_f1_baseline, 44.09, 45.41, 35.65]

# SQuAD EM Scores
squad_em_baseline = 30.00
squad_em_scheme1 = [squad_em_baseline, 32.5, 29, 18.33]
squad_em_scheme2 = [squad_em_baseline, 27.5, 31, 16.67]

# TriviaQA Data
tqa_x = [256, 512, 1024, 2048]

# TriviaQA F1 Scores: [Baseline(256), Scheme1(512), Scheme1(1024)] / [Baseline(256), Scheme2(512), Scheme2(1024)]
tqa_f1_baseline = 41.06
tqa_f1_scheme1 = [tqa_f1_baseline, 38.44, 41.49, 33.78]
tqa_f1_scheme2 = [tqa_f1_baseline, 39.44, 40.03, 35.0]

# TriviaQA EM Scores
tqa_em_baseline = 25.33
tqa_em_scheme1 = [tqa_em_baseline, 24.0, 27.0, 20.67]
tqa_em_scheme2 = [tqa_em_baseline, 25, 25.5, 22.0]

# ==========================================
# 3. FIGURE SETUP & PLOTTING (SEPARATE FIGURES)
# ==========================================


def style_axes(ax, x_ticks, x_range):
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Chunk Size (Tokens)", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")

    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle="--", alpha=0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # Keep dataset-specific x ranges exactly as requested.
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(5, 75)

    ax.annotate(
        "",
        xy=(1.02, 0),
        xytext=(1.0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12),
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0, 1.02),
        xytext=(0, 1.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12),
        clip_on=False,
    )


def add_dataset_legend(ax, baseline_color, dataset_prefix):
    handles, labels = ax.get_legend_handles_labels()
    baseline_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker=MARKER_BASELINE,
        markersize=8,
        color=baseline_color,
        label=f"{dataset_prefix} Baseline (F1 & EM)",
    )

    legend = ax.legend(
        handles + [baseline_handle],
        labels + [baseline_handle.get_label()],
        loc="upper right",
        frameon=True,
        framealpha=0.8,
        handlelength=3.8,
        handletextpad=0.8,
    )
    legend.get_frame().set_edgecolor("none")


def save_in_three_formats(fig, output_prefix):
    fig.savefig(f"{output_prefix}.pdf", format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", format="png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", format="svg", dpi=600, bbox_inches="tight")


def plot_single_dataset(
    x_values,
    f1_scheme1,
    f1_scheme2,
    em_scheme1,
    em_scheme2,
    color,
    dataset_prefix,
    output_prefix,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_values,
        f1_scheme1,
        color=color,
        linestyle="-",
        marker=MARKER_SCHEME1,
        markevery=slice(1, None),
        markersize=12,
        label=f"{dataset_prefix} Adaptive Adjacent Merging F1",
        zorder=3,
    )
    ax.plot(
        x_values,
        f1_scheme2,
        color=color,
        linestyle="-",
        marker=MARKER_SCHEME2,
        markevery=slice(1, None),
        markersize=8,
        label=f"{dataset_prefix} Semantic Neighbor Merging F1",
        zorder=3,
    )
    ax.plot(
        x_values,
        em_scheme1,
        color=color,
        linestyle="--",
        marker=MARKER_SCHEME1,
        markevery=slice(1, None),
        markersize=12,
        label=f"{dataset_prefix} Adaptive Adjacent Merging EM",
        zorder=3,
    )
    ax.plot(
        x_values,
        em_scheme2,
        color=color,
        linestyle="--",
        marker=MARKER_SCHEME2,
        markevery=slice(1, None),
        markersize=8,
        label=f"{dataset_prefix} Semantic Neighbor Merging EM",
        zorder=3,
    )

    baseline_points = [
        (x_values[0], f1_scheme1[0]),
        (x_values[0], f1_scheme2[0]),
        (x_values[0], em_scheme1[0]),
        (x_values[0], em_scheme2[0]),
    ]
    for x_base, y_base in baseline_points:
        ax.plot(
            x_base,
            y_base,
            linestyle="None",
            marker=MARKER_BASELINE,
            markersize=8,
            color=color,
            zorder=4,
        )

    # Add a small margin so boundary markers are not clipped by the axis limits.
    x_range = (x_values[0] * 0.88, x_values[-1] * 1.12)
    style_axes(ax, x_ticks=x_values, x_range=x_range)
    add_dataset_legend(ax, baseline_color=color, dataset_prefix=dataset_prefix)

    fig.tight_layout()
    save_in_three_formats(fig, output_prefix)

    return fig


# SQuAD figure: x-axis range 64 to 512
plot_single_dataset(
    x_values=squad_x,
    f1_scheme1=squad_f1_scheme1,
    f1_scheme2=squad_f1_scheme2,
    em_scheme1=squad_em_scheme1,
    em_scheme2=squad_em_scheme2,
    color=COLOR_SQUAD,
    dataset_prefix="SQuAD",
    output_prefix="squad_chunk_performance",
)

# TQA figure: x-axis range 256 to 2048
plot_single_dataset(
    x_values=tqa_x,
    f1_scheme1=tqa_f1_scheme1,
    f1_scheme2=tqa_f1_scheme2,
    em_scheme1=tqa_em_scheme1,
    em_scheme2=tqa_em_scheme2,
    color=COLOR_TQA,
    dataset_prefix="TQA",
    output_prefix="tqa_chunk_performance",
)

plt.show()
