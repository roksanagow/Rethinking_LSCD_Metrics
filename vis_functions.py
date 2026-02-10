import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_any_effect(df, factor1, factor2):
    """
    Computes mean, std, and count of Spearman per factor1 × factor2.
    """
    return (
        df
        .groupby([factor1, factor2])["spearman"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values([factor1, "mean"], ascending=[True, False])
    )

def _dot3(x: float, fmt: str = ".3f") -> str:
    s = format(x, fmt)  # e.g. "-0.123" or "0.045" or "1.234"
    if s.startswith("-0."):
        return "-." + s[3:]   # "-0.123" -> "-.123"
    if s.startswith("0."):
        return "." + s[2:]    # "0.123"  -> ".123"
    return s  

def plot_effect_heatmap(
    effect_df: pd.DataFrame,
    *,
    col_col: str | None = None,
    row_col: str = "language",
    annotate: bool = True,
    include_std: bool = True,
    fmt: str = ".3f",
    figsize: tuple | None = None,
    title: str | None = None,
    sort_rows: bool = False,
    sort_cols: bool = False,
    vmin=None,
    vmax=None,
    col_order: list | None = None,
    print_table: bool = False,
    spaces: list = ['full','def','pca','rand'],
    metrics: list = ["apd", "prt", "amd", "samd"],
):
    """
    Heatmap with:
      mean
      (std)
    annotations. Counts are printed, not shown.
    """

    df = effect_df.copy()

    # Infer factor column
    if col_col is None:
        candidates = [c for c in df.columns if c not in {row_col, "mean", "std", "count"}]
        if len(candidates) == 1:
            col_col = candidates[0]
        else:
            raise ValueError(f"Please specify col_col. Candidates: {candidates}")

    # Print count diagnostics
    if "count" in df.columns:
        print_count_summary(df, row_col, col_col)

    mean_mat = df.pivot_table(index=row_col, columns=col_col, values="mean", sort=False)
    std_mat = df.pivot_table(index=row_col, columns=col_col, values="std", sort=False)
    # Align std matrix to mean matrix so missing rows/cols become NaN (no KeyError)
    std_mat = std_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)


    if sort_rows:
        order = metrics
        # order = mean_mat.mean(axis=1).sort_values(ascending=False).index
        mean_mat = mean_mat.loc[order]
        std_mat = std_mat.loc[order]

    if sort_cols:
        # order = mean_mat.mean(axis=0).sort_values(ascending=False).index
        order = spaces
        mean_mat = mean_mat[order]
        std_mat = std_mat[order]
        if col_order is not None:
            mean_mat = mean_mat[col_order]
            std_mat = std_mat[col_order]
        elif col_col == 'metric_space':
            # sort by space, then metric
            metrics = ['apd','prt','amd','samd']
            ordered_cols = []
            for space in spaces:
                for metric in metrics:
                    col_name = f"{metric}_{space}"
                    if col_name in mean_mat.columns:
                        ordered_cols.append(col_name)
            mean_mat = mean_mat[ordered_cols]
            std_mat = std_mat[ordered_cols]

    # order mean and std matrices by spaces (for columns)
    if col_col == 'space':
        order = spaces  
        mean_mat = mean_mat[order]
        std_mat = std_mat[order]


    # Build annotation matrix
    ann = None
    if annotate:
        ann = mean_mat.copy().astype(object)
        for r in mean_mat.index:
            for c in mean_mat.columns:
                m = mean_mat.loc[r, c]
                if pd.isna(m):
                    ann.loc[r, c] = ""
                elif include_std and r != "Average" and c != "Average":
                    s = std_mat.loc[r, c]
                    if pd.isna(s):
                        ann.loc[r, c] = _dot3(m, fmt)
                    else:
                        ann.loc[r, c] = f"{_dot3(m, fmt)}\n({_dot3(s, fmt)})"
                else:
                    ann.loc[r, c] = _dot3(m, fmt)

    if figsize is None:
        figsize = (max(6, 1 + 0.8 * mean_mat.shape[1]),
                   max(4, 1 + 0.6 * mean_mat.shape[0]))

    fig, ax = plt.subplots(figsize=figsize)


    sns.heatmap(
        mean_mat,
        ax=ax,
        cmap="viridis",
        annot=ann if annotate else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": ""},
        linewidths=0.5,
        linecolor="white"
    )

    if title is None:
        title = f"Mean by {row_col} × {col_col}"
    ax.set_title(title)

    ax.set_xlabel(col_col.capitalize())
    ax.set_ylabel(row_col.capitalize())
    ax.set_xticklabels([lbl.get_text().upper() for lbl in ax.get_xticklabels()], rotation=0, ha="center")
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()], rotation=0)

    plt.tight_layout()
    plt.show()

    if print_table:
        # round mean and std matrices for better readability
        mean_mat = mean_mat.round(3)
        std_mat = std_mat.round(3)
        print("\nMean matrix:")
        print(mean_mat)
        print("\nStd matrix:")
        print(std_mat)

    # add average row to std_mat
    avg_std_row = std_mat.mean(axis=0)
    avg_std_row.name = "Average"
    std_mat = pd.concat([std_mat, avg_std_row.to_frame().T])

    return mean_mat, std_mat

def plot_effect_heatmap_hierarchical(
    effect_df: pd.DataFrame,
    *,
    col_col: str | None = None,
    row_col: str = "language",
    annotate: bool = True,
    include_std: bool = True,
    fmt: str = ".3f",
    figsize: tuple | None = None,
    title: str | None = None,
    sort_rows: bool = False,
    sort_cols: bool = False,
    vmin=None,
    vmax=None,
    col_order: list | None = None,
    print_table: bool = False,
    spaces: list = ['full','def','pca','rand'],
    metrics: list = ["apd", "prt","amd", "samd"],
    tick_fontsize: int = 12,
):
    """
    Heatmap with:
      mean
      (std)
    annotations. Counts are printed, not shown.
    """

    df = effect_df.copy()

    # Infer factor column
    if col_col is None:
        candidates = [c for c in df.columns if c not in {row_col, "mean", "std", "count"}]
        if len(candidates) == 1:
            col_col = candidates[0]
        else:
            raise ValueError(f"Please specify col_col. Candidates: {candidates}")

    # Print count diagnostics
    if "count" in df.columns:
        print_count_summary(df, row_col, col_col)

    mean_mat = df.pivot_table(index=row_col, columns=col_col, values="mean")
    std_mat = df.pivot_table(index=row_col, columns=col_col, values="std")
    std_mat = std_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)

    if sort_rows:
        order = mean_mat.mean(axis=1).sort_values(ascending=False).index
        mean_mat = mean_mat.loc[order]
        std_mat = std_mat.loc[order]

    if sort_cols:
        # order = mean_mat.mean(axis=0).sort_values(ascending=False).index
        # order according to spaces
        order = []
        for space in (spaces or []):
            for metric in metrics:
                col_name = f"{metric}_{space}"
                if col_name in mean_mat.columns:
                    order.append(col_name)
        mean_mat = mean_mat[order]
        std_mat = std_mat[order]

    # Custom ordering for metric_space
    if col_order is not None:
        mean_mat = mean_mat[col_order]
        std_mat = std_mat[col_order]
    elif col_col == "metric_space":
        ordered_cols = []
        for space in (spaces or []):
            for metric in metrics:
                col_name = f"{metric}_{space}"
                if col_name in mean_mat.columns:
                    ordered_cols.append(col_name)
        # fall back if spaces not provided or columns missing
        if ordered_cols:
            mean_mat = mean_mat[ordered_cols]
            std_mat = std_mat[ordered_cols]

    # Add averages (mean only)
    # mean_mat["Average"] = mean_mat.mean(axis=1)
    avg_row = mean_mat.mean(axis=0)
    avg_row.name = "Average"
    mean_mat = pd.concat([mean_mat, avg_row.to_frame().T])

    # Build annotation matrix
    ann = None
    if annotate:
        ann = mean_mat.copy().astype(object)
        for r in mean_mat.index:
            for c in mean_mat.columns:
                m = mean_mat.loc[r, c]
                if pd.isna(m):
                    ann.loc[r, c] = ""
                elif include_std and r != "Average" and c != "Average":
                    s = std_mat.loc[r, c]
                    if pd.isna(s):
                        ann.loc[r, c] = _dot3(m, fmt)
                    else:
                        ann.loc[r, c] = f"{_dot3(m, fmt)}\n({_dot3(s, fmt)})"
                else:
                    ann.loc[r, c] = _dot3(m, fmt)

    if figsize is None:
        figsize = (max(6, 1 + 0.8 * mean_mat.shape[1]),
                   max(4, 1 + 0.6 * mean_mat.shape[0]))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        mean_mat,
        ax=ax,
        cmap="viridis",
        annot=ann if annotate else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": ""},
        linewidths=0.5,
        linecolor="white"
        )
    
    # --- vertical separators between spaces (every 3 columns) ---
    group_size = len(metrics)
    n_cols = mean_mat.shape[1]

    for x in range(group_size, n_cols, group_size):
            ax.axvline(
                x,
                color="black",
                linewidth=1,
                ymin=-0.1,   # extend below heatmap into tick-label area
                ymax=1.0,
                clip_on=False
            )


    if title is None:
        title = f"Mean by {row_col} × {col_col}"
    ax.set_title(title)

    ax.set_xlabel(col_col.upper())
    ax.set_ylabel(row_col.capitalize(), fontsize=tick_fontsize + 2)

        # ---------------------------
    # Hierarchical x tick labels
    # ---------------------------
    cols = list(mean_mat.columns)

    def _metric_space(name: str):
        # expects "metric_space" like "apd_full" (space may contain underscores)
        if name == "Average":
            return ("Average", "")
        
        parts2 = [s for s in spaces if s in name][0]
        # other part is what remains when space is removed
        parts1 = name.replace(parts2, "").rstrip("_")

        return (parts1, parts2)

    if col_col == "metric_space":
        # Per-column metric labels (line just under heatmap)
        metric_labels = []
        space_labels = []
        for c in cols:
            m, s = _metric_space(c)
            metric_labels.append(m.upper() if m != "Average" else "Average")
            space_labels.append(s)

        # Line 1: METRICS at each column center
        ax.set_xticklabels(metric_labels, rotation=45, ha="center", fontsize=tick_fontsize)

        ax.tick_params(axis="x", length=0, pad=2)  # no tick marks

        # Line 2: SPACES, one label per group of 3 metrics (hard-coded)
        group_size = len(metrics)
        ncols = len(cols)

        # Determine how many full groups we have (exclude trailing "Average" if present)
        has_avg = (ncols > 0 and cols[-1] == "Average")
        n_metric_cols = ncols - (1 if has_avg else 0)
        n_groups = n_metric_cols // group_size

        # Compute tick positions at group centers in heatmap coordinates.
        # Heatmap column centers are at 0.5, 1.5, ..., so group center for group g is:
        # (g*group_size + (group_size-1)/2) + 0.5
        # group_centers = [g * group_size + (group_size - 1) / 2 + 0.5 for g in range(n_groups)]
        group_centers = []
        for g in range(n_groups):
            start = g * group_size
            end = min((g + 1) * group_size, n_metric_cols)
            center = (start + (end - 1)) / 2 + 0.5
            group_centers.append(center)

        # group_names = []
        # for g in range(n_groups):
        #     # take the space name from the first col in the group
        #     _, s = _metric_space(cols[g * group_size])
        #     group_names.append(s.replace("_", " ").upper())
        group_names = []
        for g in range(n_groups):
            start = g * group_size
            if start >= n_metric_cols:
                break
            _, s = _metric_space(cols[start])
            group_names.append(s.replace("_", " ").upper())


        # Optional: include Average as its own group label (blank or "Average")
        # Here: leave it blank to avoid redundancy
        if has_avg:
            group_centers.append(n_metric_cols + 0.5)
            group_names.append("")

        # Create second x-axis for grouped space labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_names, rotation=0, ha="center", fontsize=tick_fontsize)
        ax2.tick_params(axis="x", length=0, pad=30 + tick_fontsize)


        # Remove spines so there's no line between the two label rows
        for side in ["top", "bottom", "left", "right"]:
            ax2.spines[side].set_visible(False)

        # Optional: remove axis labels since hierarchy is self-explanatory
        ax.set_xlabel("")
        ax2.set_xlabel("")
    else:
        # Non-hierarchical case: uppercase tick labels except "Average"
        base = [lbl.get_text() for lbl in ax.get_xticklabels()]
        base = [t if t == "Average" else t.upper() for t in base]
        ax.set_xticklabels(base, rotation=0, ha="right")

    # Y ticks: Capitalise (not full upper)
    ax.set_yticklabels(
    [label.get_text().replace("_", " ").title() for label in ax.get_yticklabels()],
    rotation=0,
    fontsize=tick_fontsize
)

    plt.tight_layout()
    plt.show()

    if print_table:
        mean_mat_round = mean_mat.round(3)
        std_mat_round = std_mat.round(3)
        print("\nMean matrix:")
        print(mean_mat_round)
        print("\nStd matrix:")
        print(std_mat_round)

    # add average row to std_mat
    avg_std_row = std_mat.mean(axis=0)
    avg_std_row.name = "Average"
    std_mat = pd.concat([std_mat, avg_std_row.to_frame().T])

    return mean_mat, std_mat

def print_count_summary(effect_df, row_col, col_col):
    counts = effect_df.groupby([row_col, col_col])["count"].mean()
    print(
        f"[Count summary] mean={counts.mean():.2f}, "
        f"min={counts.min():.0f}, max={counts.max():.0f}"
    )

def process_df(df, encoders = ['xll','rembert','xlmr','multilingual-e5','mmbert','monolingual_model']):
    # change values in the encoders column in the df not in encoders list to 'monolingual_model'
    df['encoder'] = df.apply(lambda row: 'monolingual_model' if row['encoder'] not in encoders else row['encoder'], axis=1)
    # add a metric_space column
    df['metric_space'] = df['metric'] + '_' + df['space']

    # Ensure categorical typing (important for stats)
    categorical_cols = [
        "language", "encoder", "definition_gen_model",
        "metric", "space", "metric_space"
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    df["spearman"] = df["spearman"].astype(float)
    return df