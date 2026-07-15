#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PKG_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_DIR / "data"
OUT_DIR = PKG_DIR / "output"

PATTERNS = ["Pattern 1", "Pattern 2", "Pattern 3"]
PATTERN_COLORS = {
    "Pattern 1": "#1F4E79",
    "Pattern 2": "#C55A11",
    "Pattern 3": "#2E8B57",
}
PALE_COLORS = {
    "Pattern 1": "#9DB8CE",
    "Pattern 2": "#E4A56F",
    "Pattern 3": "#A7CFB6",
}
SCENARIO_COLORS = {"ssp126": "#2C7FB8", "ssp585": "#D94A38"}
LINE_DISPLAY_NAMES = {
    "长永线（455）": "Zhangyong Xian Line",
    "壤武线（457）": "Rangwu Xian Line",
    "白普线（455）": "Baipu Xian Line",
}
BIVARIATE_LINE_COLORS = np.array(
    [
        ["#E8E8E8", "#E4ACAC", "#C85A5A"],
        ["#B0D5DF", "#AD9EA5", "#985356"],
        ["#64ACBE", "#627F8C", "#574249"],
    ]
)
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_red_blue", ["#1E88E5", "#7B1FA2", "#FF0052"])
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "temp_muted", ["#F7FAFC", "#DCEAF2", "#F3D0B9", "#D36D5C", "#7F1D35"], N=256
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def pattern_short(pattern: str) -> str:
    return pattern.replace("Pattern ", "P")


def line_display_name(line_name: str) -> str:
    return LINE_DISPLAY_NAMES.get(line_name, line_name)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#111827",
        zorder=50,
    )


def panel_label_inside(ax: plt.Axes, label: str, *, x: float = 0.025, y: float = 0.96) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#111827",
        zorder=50,
    )


def save_figure(fig: plt.Figure, stem: str, output_dir: Path = OUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)


def cleanup_appledouble(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()


def read_lines(target_crs=None) -> gpd.GeoDataFrame:
    lines = gpd.read_file(DATA_DIR / "common/line_geometries.gpkg")
    if target_crs is not None:
        lines = lines.to_crs(target_crs)
    return lines


def read_boundary(target_crs=None) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(DATA_DIR / "common/wanzhou_district_boundary.gpkg")
    if target_crs is not None:
        boundary = boundary.to_crs(target_crs)
    return boundary


def map_limits(boundary: gpd.GeoDataFrame, pad_frac: float = 0.035) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = boundary.total_bounds
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def setup_map_axis(ax: plt.Axes, boundary: gpd.GeoDataFrame, *, show_axis: bool = False) -> None:
    xmin, xmax, ymin, ymax = map_limits(boundary)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect((ymax - ymin) / (xmax - xmin))
    if not show_axis:
        ax.set_axis_off()


def plot_base_boundary(ax: plt.Axes, boundary: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#F7F8FA", edgecolor="#303740", linewidth=0.58, zorder=0)
    boundary.boundary.plot(ax=ax, color="#303740", linewidth=0.58, zorder=6)


def add_horizontal_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str,
    *,
    width: str = "58%",
    y: float = -0.10,
) -> None:
    cax = ax.inset_axes([(1 - float(width.strip("%")) / 100) / 2, y, float(width.strip("%")) / 100, 0.045])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=5.5, length=1.5, pad=1.0)
    cbar.outline.set_linewidth(0.35)
    if label:
        cbar.set_label(label, fontsize=5.8, labelpad=1.0)


def draw_pattern_line_map(ax: plt.Axes, lines: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> None:
    plot_base_boundary(ax, boundary)
    for pattern in PATTERNS:
        sub = lines[lines["pattern_label"].eq(pattern)]
        if not sub.empty:
            sub.plot(ax=ax, color=PATTERN_COLORS[pattern], linewidth=0.72, alpha=0.95, zorder=10)
    setup_map_axis(ax, boundary)


def draw_value_line_map(
    fig: plt.Figure,
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    column: str,
    *,
    cmap,
    norm=None,
    label: str = "",
    linewidth: float = 0.82,
    boundary: gpd.GeoDataFrame | None = None,
) -> None:
    if boundary is None:
        boundary = read_boundary(gdf.crs)
    plot_base_boundary(ax, boundary)
    gdf.plot(ax=ax, column=column, cmap=cmap, norm=norm, linewidth=linewidth, zorder=8)
    setup_map_axis(ax, boundary)
    sm = mpl.cm.ScalarMappable(norm=norm or Normalize(vmin=gdf[column].min(), vmax=gdf[column].max()), cmap=cmap)
    add_horizontal_colorbar(fig, ax, sm, label)


def month_ticks_365() -> tuple[list[int], list[str]]:
    return [1, 60, 121, 182, 244, 305], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]


def plot_main_figure_1(output_dir: Path = OUT_DIR) -> None:
    heat_colors = plt.imread(DATA_DIR / "main_figure_1/line_daily_heatmap_colors.png")
    pattern_strip = plt.imread(DATA_DIR / "main_figure_1/line_daily_pattern_strip.png")
    monthly = pd.read_csv(DATA_DIR / "main_figure_1/line_monthly_normalized_profiles.csv")
    inventory = pd.read_csv(DATA_DIR / "main_figure_1/pattern_inventory.csv")
    lines = read_lines()
    boundary = read_boundary(lines.crs)

    if heat_colors.ndim != 3 or heat_colors.shape[1] != 365 or heat_colors.shape[2] not in (3, 4):
        raise ValueError("Main Figure 1 heatmap colors must be an RGB(A) image with 365 columns.")
    if pattern_strip.ndim != 3 or pattern_strip.shape[:2] != (1, heat_colors.shape[0]):
        raise ValueError("Main Figure 1 pattern strip must contain one RGB(A) color per heatmap row.")

    fig = plt.figure(figsize=(183 / 25.4, 128 / 25.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.26, 0.82], width_ratios=[1, 1, 1, 1.35], hspace=0.28, wspace=0.34)
    ax_heat = fig.add_subplot(gs[0, :])
    ax_heat.imshow(heat_colors, aspect="auto", interpolation="nearest")
    ticks, labels = month_ticks_365()
    ax_heat.set_xticks([t - 1 for t in ticks], labels)
    ax_heat.set_yticks([])
    ax_heat.set_ylabel("Lines")
    ax_heat.set_xlim(0, 364)
    panel_label_inside(ax_heat, "a", x=-0.018, y=1.01)
    cax = ax_heat.inset_axes([1.018, 0.02, 0.018, 0.96])
    heat_scale = mpl.cm.ScalarMappable(norm=Normalize(vmin=0.60, vmax=1.72), cmap="RdBu_r")
    heat_scale.set_array([])
    cbar = fig.colorbar(heat_scale, cax=cax)
    cbar.set_label("Normalized normal load", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.2, length=2)

    strip = ax_heat.inset_axes([0.0, 1.006, 1.0, 0.018])
    strip.imshow(pattern_strip, aspect="auto", interpolation="nearest")
    strip.set_axis_off()

    for idx, pattern in enumerate(PATTERNS):
        ax = fig.add_subplot(gs[1, idx])
        sub = monthly[monthly["pattern"].eq(pattern)]
        q = sub.groupby("month")["normal_index"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
        mean = sub.groupby("month")["normal_index"].mean()
        months = q.index.to_numpy()
        ax.fill_between(months, q[0.10], q[0.90], color=PATTERN_COLORS[pattern], alpha=0.17, lw=0, label="P10-P90")
        ax.fill_between(months, q[0.25], q[0.75], color=PATTERN_COLORS[pattern], alpha=0.34, lw=0, label="P25-P75")
        ax.plot(months, mean, color=PATTERN_COLORS[pattern], lw=1.55, label="Mean")
        ax.axhline(1.0, color="#7A7F87", lw=0.6, ls="--")
        count = int(inventory.loc[inventory["pattern_label"].eq(pattern), "n_lines"].iloc[0])
        ax.text(0.03, 0.93, f"{pattern}  n={count}", color=PATTERN_COLORS[pattern], transform=ax.transAxes, ha="left", va="top", fontsize=7.4, fontweight="bold")
        ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("Normalized normal load")
        ax.grid(color="#E5E7EB", lw=0.45)
        panel_label(ax, chr(ord("b") + idx), x=-0.08, y=1.03)
        if pattern == "Pattern 3":
            ax.set_ylim(0.55, max(4.7, float(q[0.90].max()) * 1.05))
        else:
            ax.set_ylim(0.55, max(1.75, float(q[0.90].max()) * 1.05))

    ax_map = fig.add_subplot(gs[1, 3])
    draw_pattern_line_map(ax_map, lines, boundary)
    handles = [Line2D([0], [0], color=PATTERN_COLORS[p], lw=1.6, label=p) for p in PATTERNS]
    ax_map.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.00, -0.06), fontsize=6.2)
    panel_label(ax_map, "e", x=-0.08, y=1.03)
    save_figure(fig, "main_figure_1_operating_modes", output_dir)
    plt.close(fig)


def draw_bivariate_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    cube = ax.inset_axes([0.18, 0.18, 0.70, 0.70])
    for load_idx in range(3):
        for temp_idx in range(3):
            cube.add_patch(
                Rectangle((temp_idx, load_idx), 1, 1, facecolor=BIVARIATE_LINE_COLORS[load_idx, temp_idx], edgecolor="white", linewidth=0.8)
            )
    cube.set_xlim(0, 3)
    cube.set_ylim(0, 3)
    cube.set_aspect("equal")
    cube.set_xticks([0.5, 1.5, 2.5], ["low", "mid", "high"])
    cube.set_yticks([0.5, 1.5, 2.5], ["low", "mid", "high"])
    cube.tick_params(length=0, labelsize=6.2)
    cube.set_xlabel("Annual mean temperature", fontsize=6.4, labelpad=2)
    cube.set_ylabel("Annual mean transmission", fontsize=6.4, labelpad=2)
    for spine in cube.spines.values():
        spine.set_visible(False)


def plot_bivariate_map_panel(ax: plt.Axes, gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, year: int, pattern: str) -> None:
    plot_base_boundary(ax, boundary)
    sub = gdf[(gdf["year"].eq(year)) & (gdf["pattern_label"].eq(pattern))]
    for color, part in sub.groupby("bivariate_color"):
        part.plot(ax=ax, color=color, linewidth=0.72, alpha=0.96, zorder=10)
    setup_map_axis(ax, boundary, show_axis=False)


def plot_main_figure_2(output_dir: Path = OUT_DIR) -> None:
    lines = read_lines()
    boundary = read_boundary(lines.crs)
    annual = pd.read_csv(DATA_DIR / "main_figure_2/annual_temperature_load.csv")
    bins = pd.read_csv(DATA_DIR / "main_figure_2/temperature_response_bins.csv")
    peaks = pd.read_csv(DATA_DIR / "main_figure_2/temperature_response_peaks.csv")
    gdf = lines.merge(annual, on="line_id", how="inner", suffixes=("", "_src"))
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=lines.crs)

    years = [2021, 2022, 2023, 2024]
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    left, right = 0.075, 0.985
    col_gap = 0.018
    col_w = (right - left - 3 * col_gap) / 4
    map_h = 0.175
    map_top = 0.935
    row_gap = 0.030
    panel = 0
    for r, pattern in enumerate(PATTERNS):
        for c, year in enumerate(years):
            x0 = left + c * (col_w + col_gap)
            y0 = map_top - (r + 1) * map_h - r * row_gap
            ax = fig.add_axes([x0, y0, col_w, map_h])
            plot_bivariate_map_panel(ax, gdf, boundary, year, pattern)
            panel_label_inside(ax, chr(ord("a") + panel), x=0.02, y=0.96)
            if r == 0:
                ax.text(0.5, 1.02, str(year), transform=ax.transAxes, ha="center", va="bottom", fontsize=8.0)
            if c == 0:
                ax.text(-0.08, 0.5, pattern_short(pattern), transform=ax.transAxes, ha="right", va="center", color=PATTERN_COLORS[pattern], fontsize=8.0, fontweight="bold")
            panel += 1

    ax_resp = fig.add_axes([0.085, 0.070, 0.405, 0.270])
    for pattern in PATTERNS:
        sub = bins[bins["pattern"].eq(pattern)].sort_values("temp_c")
        ax_resp.fill_between(sub["temp_c"], sub["normalized_transmission_ci95_low"], sub["normalized_transmission_ci95_high"], color=PATTERN_COLORS[pattern], alpha=0.15, lw=0)
        ax_resp.plot(sub["temp_c"], sub["normalized_transmission_median"], color=PATTERN_COLORS[pattern], lw=1.2, label=pattern_short(pattern))
    for row in peaks.itertuples():
        ax_resp.scatter(row.temp_c, row.normalized_transmission_median, color=PATTERN_COLORS[row.pattern], s=14, zorder=8)
    ax_resp.axhline(1.0, color="#7A7F87", lw=0.55, ls="--")
    ax_resp.set_xlabel("7-d near-line mean temperature (deg C)")
    ax_resp.set_ylabel("Line-normalized transmission")
    ax_resp.grid(color="#E5E7EB", lw=0.45)
    ax_resp.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax_resp, "m")

    ax_dist = fig.add_axes([0.550, 0.070, 0.205, 0.270])
    vals = [annual.loc[annual["pattern_label"].eq(p), "annual_tmean_c"].dropna().to_numpy() for p in PATTERNS]
    parts = ax_dist.violinplot(vals, positions=np.arange(1, 4), widths=0.68, showextrema=False)
    for body, pattern in zip(parts["bodies"], PATTERNS):
        body.set_facecolor(PATTERN_COLORS[pattern])
        body.set_edgecolor("none")
        body.set_alpha(0.18)
    rng = np.random.default_rng(20260626)
    for i, (pattern, arr) in enumerate(zip(PATTERNS, vals), 1):
        ax_dist.scatter(i + rng.normal(0, 0.035, size=len(arr)), arr, s=4.5, color=PATTERN_COLORS[pattern], alpha=0.17, linewidth=0, rasterized=True)
        q25, med, q75 = np.nanpercentile(arr, [25, 50, 75])
        ax_dist.vlines(i, q25, q75, color=PATTERN_COLORS[pattern], lw=2.0)
        ax_dist.hlines(med, i - 0.14, i + 0.14, color="#FFFFFF", lw=1.1)
    ax_dist.set_xticks([1, 2, 3], [pattern_short(p) for p in PATTERNS])
    ax_dist.set_ylabel("Annual Tmean (deg C)")
    ax_dist.grid(axis="y", color="#E5E7EB", lw=0.45)
    panel_label(ax_dist, "n")

    ax_leg = fig.add_axes([0.800, 0.085, 0.175, 0.235])
    draw_bivariate_legend(ax_leg)
    save_figure(fig, "main_figure_2_temperature_response_bivariate_line_maps", output_dir)
    plt.close(fig)


def jittered_points(ax: plt.Axes, values: pd.DataFrame, metric: str) -> None:
    rng = np.random.default_rng(20260626)
    positions = np.arange(1, 4)
    arrays = [values.loc[values["pattern"].eq(p), metric].dropna().to_numpy(float) for p in PATTERNS]
    if metric == "r2_kwh":
        arrays = [np.clip(arr, -0.2, 1.0) for arr in arrays]
    parts = ax.violinplot(arrays, positions=positions - 0.12, widths=0.55, showmeans=False, showextrema=False, showmedians=False)
    for body, pattern in zip(parts["bodies"], PATTERNS):
        body.set_facecolor(PATTERN_COLORS[pattern])
        body.set_edgecolor("none")
        body.set_alpha(0.18)
    for x, pattern, arr in zip(positions, PATTERNS, arrays):
        ax.boxplot([arr], positions=[x], widths=0.22, patch_artist=True, showfliers=False, boxprops={"facecolor": "white", "edgecolor": PATTERN_COLORS[pattern], "linewidth": 0.85}, medianprops={"color": "#111827", "linewidth": 0.8}, whiskerprops={"color": "#111827", "linewidth": 0.7}, capprops={"color": "#111827", "linewidth": 0.7})
        ax.scatter(np.full(len(arr), x + 0.18) + rng.normal(0, 0.035, len(arr)), arr, s=10, color=PATTERN_COLORS[pattern], alpha=0.55, linewidth=0, rasterized=True)
        ax.text(x, 1.04, f"n={len(arr)}\nmed={np.nanmedian(arr):.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, color="#4B5563")
    ax.set_xticks(positions, [pattern_short(p) for p in PATTERNS])
    ax.grid(axis="y", color="#E5E7EB", lw=0.45)


def plot_main_figure_3(output_dir: Path = OUT_DIR) -> None:
    metrics = pd.read_csv(DATA_DIR / "main_figure_3/rf_line_time_series_metrics.csv")
    reps = pd.read_csv(DATA_DIR / "main_figure_3/rf_line_time_series_representatives.csv")
    preds = pd.read_csv(DATA_DIR / "main_figure_3/rf_representative_line_predictions.csv", parse_dates=["date"])
    lines = read_lines()
    boundary = read_boundary(lines.crs)
    holdout = metrics[metrics["split"].eq("holdout_2024")].copy()

    fig = plt.figure(figsize=(183 / 25.4, 142 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.05], hspace=0.24, wspace=0.31)
    ax_rmse = fig.add_subplot(gs[0, 0])
    jittered_points(ax_rmse, holdout, "rmse_mwh_per_day")
    ax_rmse.set_ylabel("RF held-out RMSE\n(MWh d$^{-1}$)")
    panel_label(ax_rmse, "a")

    ax_r2 = fig.add_subplot(gs[0, 1])
    jittered_points(ax_r2, holdout, "r2_kwh")
    ax_r2.axhline(0, color="#9CA3AF", ls="--", lw=0.6)
    ax_r2.set_ylim(-0.23, 1.05)
    ax_r2.set_ylabel("RF held-out R$^2$")
    panel_label(ax_r2, "b")

    ts_gs = gs[1, 0].subgridspec(3, 1, hspace=0.08)
    ts_axes = [fig.add_subplot(ts_gs[i, 0]) for i in range(3)]
    for ax, pattern in zip(ts_axes, PATTERNS):
        line = reps.loc[reps["pattern"].eq(pattern), "line_name"].iloc[0]
        sub = preds[preds["line_name"].eq(line)].sort_values("date").copy()
        ax.plot(sub["date"], sub["observed_kwh"] / 1000.0, color="#9AA0A8", lw=0.78, label="Observed")
        ax.plot(sub["date"], sub["predicted_kwh"] / 1000.0, color=PATTERN_COLORS[pattern], lw=1.10, label="Predicted")
        r2 = reps.loc[reps["line_name"].eq(line), "r2_kwh"].iloc[0]
        ax.text(0.02, 0.84, f"{line_display_name(line)}\nR$^2$={r2:.2f}", transform=ax.transAxes, ha="left", va="top", color=PATTERN_COLORS[pattern], fontsize=6.8, fontweight="bold")
        ax.grid(color="#E5E7EB", lw=0.40)
        if ax is not ts_axes[-1]:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Held-out 2024")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if ax is ts_axes[1]:
            ax.set_ylabel("Line transmission (MWh d$^{-1}$)")
    ts_axes[0].legend(frameon=False, loc="upper right")
    panel_label(ts_axes[0], "c")

    ax_map = fig.add_subplot(gs[1, 1])
    plot_base_boundary(ax_map, boundary)
    for pattern in PATTERNS:
        read = lines[lines["pattern_label"].eq(pattern)]
        read.plot(ax=ax_map, color=PALE_COLORS[pattern], linewidth=0.40, alpha=0.68, zorder=5)
    for pattern in PATTERNS:
        line = reps.loc[reps["pattern"].eq(pattern), "line_name"].iloc[0]
        lines[lines["line_id"].eq(line)].plot(ax=ax_map, color=PATTERN_COLORS[pattern], linewidth=2.2, zorder=12)
    setup_map_axis(ax_map, boundary)
    handles = [Line2D([0], [0], color=PALE_COLORS[p], lw=1.4, label=f"{pattern_short(p)} other") for p in PATTERNS]
    handles += [
        Line2D([0], [0], color=PATTERN_COLORS[p], lw=2.2, label=line_display_name(reps.loc[reps["pattern"].eq(p), "line_name"].iloc[0]))
        for p in PATTERNS
    ]
    ax_map.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=5.7)
    panel_label(ax_map, "d", x=-0.08)
    save_figure(fig, "main_figure_3_rf_model_performance", output_dir)
    plt.close(fig)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.blake2s(text.encode("utf-8"), digest_size=4).hexdigest(), 16)


def beeswarm_offsets(values: np.ndarray, *, seed: int, width: float = 0.38, bins: int = 52) -> np.ndarray:
    finite = np.isfinite(values)
    offsets = np.zeros(len(values), dtype=float)
    if finite.sum() < 2:
        return offsets
    vals = values[finite]
    edges = np.linspace(np.nanmin(vals), np.nanmax(vals), bins + 1)
    if np.allclose(edges[0], edges[-1]):
        return offsets
    bin_id = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    counts = np.bincount(bin_id, minlength=bins).astype(float)
    density_width = width * np.sqrt(counts[bin_id] / max(counts.max(), 1.0))
    rng = np.random.default_rng(seed)
    local = rng.normal(0, density_width / 1.75, len(vals)) + rng.uniform(-density_width, density_width, len(vals)) * 0.28
    offsets[np.where(finite)[0]] = np.clip(local, -width, width)
    return offsets


def top_features(summary: pd.DataFrame, pattern: str, n: int = 15) -> list[str]:
    sub = summary[summary["pattern"].eq(pattern)].sort_values("mean_abs_shap", ascending=False)
    return sub["display_feature"].head(n).tolist()


def draw_shap_summary(
    ax: plt.Axes,
    shap: pd.DataFrame,
    feature_values: pd.DataFrame,
    summary: pd.DataFrame,
    pattern: str,
    features: list[str],
    xlim: tuple[float, float],
) -> None:
    idx = shap.index[shap["pattern"].eq(pattern)].to_numpy()
    ypos = np.arange(len(features))[::-1]
    smry = summary[summary["pattern"].eq(pattern)].set_index("display_feature")
    bar_vals = np.array([smry.loc[f, "mean_abs_shap"] if f in smry.index else 0 for f in features], dtype=float)
    span = xlim[1] - xlim[0]
    vmax = max(float(np.nanmax(bar_vals)), 1e-6)
    for y, val in zip(ypos, bar_vals):
        ax.barh(y, span * 0.92 * val / vmax, left=xlim[0], height=0.72, color=PATTERN_COLORS[pattern], alpha=0.11, edgecolor="none", zorder=0)
    for y, feature in zip(ypos, features):
        x = shap.loc[idx, feature].to_numpy(dtype=float)
        if feature in feature_values.columns:
            color_value = feature_values.loc[idx, feature].to_numpy(dtype=float)
        else:
            color_value = np.full(len(x), np.nan)
        colors = SHAP_CMAP(np.clip(color_value, 0, 1))
        colors[~np.isfinite(color_value)] = mpl.colors.to_rgba("#8C98A4", 0.45)
        seed = stable_seed(pattern, feature)
        yjit = beeswarm_offsets(x, seed=seed)
        rng = np.random.default_rng(seed + 17)
        ax.scatter(x + rng.normal(0, (xlim[1] - xlim[0]) * 0.0012, len(x)), y + yjit, s=6.4, c=colors, alpha=0.68, linewidth=0, rasterized=True, zorder=5)
    ax.axvline(0, color="#687386", lw=0.65)
    ax.set_xlim(*xlim)
    ax.set_yticks(ypos, features)
    ax.grid(axis="x", color="#E5E7EB", lw=0.45)


def draw_shap_decision(ax: plt.Axes, shap: pd.DataFrame, meta: pd.DataFrame, pattern: str, features: list[str], xlim: tuple[float, float], norm: Normalize) -> None:
    sub = meta[meta["pattern"].eq(pattern)].copy().sort_values("pred_log1p_kwh")
    if len(sub) > 85:
        sub = sub.iloc[np.linspace(0, len(sub) - 1, 85).round().astype(int)]
    bottom_to_top = list(reversed([f for f in features if f in shap.columns]))
    ypos = np.arange(len(bottom_to_top))
    for idx, row in sub.iterrows():
        values = shap.loc[idx, bottom_to_top].to_numpy(dtype=float)
        curve = np.cumsum(values)
        output = float(row["pred_log1p_kwh"] - row["base_value"])
        ax.plot(curve, ypos, color=SHAP_CMAP(norm(output)), alpha=0.45, lw=0.56, zorder=2)
    for y in ypos:
        ax.axhline(y, color="#D8DEE8", lw=0.42, ls=(0, (1.2, 2.2)), zorder=0)
    ax.axvline(0, color="#C8CDD5", lw=0.78, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, len(bottom_to_top) - 0.5)
    ax.set_yticks(ypos, [])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_ticks_position("both")
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_linewidth(0.65)
    ax.spines["top"].set_color("#424A57")
    ax.tick_params(axis="x", top=True, labeltop=True, length=2.5, pad=1.4)


def plot_main_figure_4(output_dir: Path = OUT_DIR) -> None:
    shap = pd.read_csv(DATA_DIR / "main_figure_4/shap_display_values.csv")
    feature_values = pd.read_csv(DATA_DIR / "main_figure_4/shap_feature_values.csv")
    meta = pd.read_csv(DATA_DIR / "main_figure_4/shap_sample_metadata.csv")
    summary = pd.read_csv(DATA_DIR / "main_figure_4/shap_feature_summary.csv")
    features_by_pattern = {pattern: top_features(summary, pattern, 15) for pattern in PATTERNS}
    abs_vals = []
    for pattern, features in features_by_pattern.items():
        for feature in features:
            if feature in shap.columns:
                vals = shap.loc[shap["pattern"].eq(pattern), feature].to_numpy(float)
                abs_vals.append(np.abs(vals[np.isfinite(vals)]))
    all_abs = np.concatenate([v for v in abs_vals if len(v)]) if abs_vals else np.array([1.0])
    xmax = max(0.25, float(np.nanquantile(all_abs, 0.995)) * 1.08)
    shap_xlim = (-xmax, xmax)
    output_vals = meta["pred_log1p_kwh"].to_numpy(float) - meta["base_value"].to_numpy(float)
    bound = max(0.5, float(np.nanquantile(np.abs(output_vals[np.isfinite(output_vals)]), 0.995)) * 1.08)
    decision_xlim = (-bound, bound)
    decision_norm = Normalize(vmin=-bound, vmax=bound)

    fig = plt.figure(figsize=(183 / 25.4, 218 / 25.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.62], hspace=0.17, wspace=0.12)
    labels = [("a", "b"), ("c", "d"), ("e", "f")]
    for row, pattern in enumerate(PATTERNS):
        ax_l = fig.add_subplot(gs[row, 0])
        ax_r = fig.add_subplot(gs[row, 1])
        features = features_by_pattern[pattern]
        draw_shap_summary(ax_l, shap, feature_values, summary, pattern, features, shap_xlim)
        if row == 2:
            ax_l.set_xlabel("SHAP value (log1p kWh)")
        else:
            ax_l.tick_params(labelbottom=False)
        cax = ax_l.inset_axes([0.775, 0.075, 0.145, 0.045])
        sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=SHAP_CMAP)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(labelsize=5.0, length=1.5, pad=0.5)
        cbar.set_label("Feature value", fontsize=5.2, labelpad=1.0)
        cbar.outline.set_linewidth(0.22)
        panel_label_inside(ax_l, labels[row][0], x=0.012, y=0.98)

        draw_shap_decision(ax_r, shap, meta, pattern, features, decision_xlim, decision_norm)
        if row == 2:
            ax_r.set_xlabel("Model output value (relative to expected)")
        else:
            ax_r.tick_params(labelbottom=False)
        ax_r.text(-0.08, 0.50, pattern_short(pattern), transform=ax_r.transAxes, ha="right", va="center", fontsize=8.2, color=PATTERN_COLORS[pattern], fontweight="bold")
        panel_label_inside(ax_r, labels[row][1], x=0.012, y=0.98)
    save_figure(fig, "main_figure_4_rf_shap_pattern_mechanism", output_dir)
    plt.close(fig)


def plot_main_figure_5(output_dir: Path = OUT_DIR) -> None:
    lines = read_lines()
    boundary = read_boundary(lines.crs)
    line_period = pd.read_csv(DATA_DIR / "main_figure_5/line_period_ensemble.csv")
    annual_mode = pd.read_csv(DATA_DIR / "main_figure_5/annual_mode_ensemble.csv")
    annual_temp = pd.read_csv(DATA_DIR / "main_figure_5/annual_temperature_ensemble.csv")
    map_df = lines.merge(line_period, left_on="line_id", right_on="line_name", how="inner")
    map_gdf = gpd.GeoDataFrame(map_df, geometry="geometry", crs=lines.crs)
    periods = ["2030s", "2050s", "2090s"]
    scenarios = ["ssp126", "ssp585"]
    vmax = float(np.nanpercentile(map_gdf["annual_change_pct_mean"], 99))
    norm = Normalize(vmin=0, vmax=max(vmax, 1.0))

    fig = plt.figure(figsize=(183 / 25.4, 248 / 25.4))
    outer = fig.add_gridspec(
        8,
        2,
        height_ratios=[0.88, 0.88, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58],
        hspace=0.28,
        wspace=0.12,
    )
    top = outer[:2, :].subgridspec(2, 3, hspace=0.16, wspace=0.10)
    label_ord = iter("abcdef")
    for r, scenario in enumerate(scenarios):
        for c, period in enumerate(periods):
            ax = fig.add_subplot(top[r, c])
            sub = map_gdf[(map_gdf["scenario"].eq(scenario)) & (map_gdf["period"].eq(period))]
            plot_base_boundary(ax, boundary)
            sub.plot(ax=ax, column="annual_change_pct_mean", cmap="Spectral_r", norm=norm, linewidth=0.70, zorder=8)
            setup_map_axis(ax, boundary)
            panel_label_inside(ax, next(label_ord))
            if r == 0:
                ax.text(0.5, 1.02, period, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.0)
            if c == 0:
                ax.text(-0.08, 0.50, scenario.upper(), transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=8.0)
    cax = fig.add_axes([0.925, 0.765, 0.012, 0.150])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="Spectral_r")
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Annual line-transmission\nincrease rate (%)", fontsize=6.2, labelpad=4)
    cb.ax.tick_params(labelsize=5.8, length=1.8, pad=1.5)
    cb.outline.set_linewidth(0.35)

    def annual_stack(row_start: int, df: pd.DataFrame, ycols: tuple[str, str, str], ylabel: str, labels: tuple[str, str]) -> None:
        for col, scenario in enumerate(scenarios):
            for i, pattern in enumerate(PATTERNS):
                ax = fig.add_subplot(outer[row_start + i, col])
                sub = df[(df["scenario"].eq(scenario)) & (df["pattern"].eq(pattern))].sort_values("year")
                mean, p10, p90 = ycols
                ax.fill_between(sub["year"], sub[p10], sub[p90], color=PATTERN_COLORS[pattern], alpha=0.14, lw=0)
                ax.plot(sub["year"], sub[mean], color=PATTERN_COLORS[pattern], lw=1.25)
                ax.axhline(0, color="#9CA3AF", lw=0.55)
                ax.grid(color="#E5E7EB", lw=0.40)
                ax.set_xlim(2026, 2100)
                ax.text(0.98, 0.82, pattern_short(pattern), transform=ax.transAxes, ha="right", va="top", color=PATTERN_COLORS[pattern], fontsize=7.2, fontweight="bold")
                if col == 0 and i == 1:
                    ax.set_ylabel(ylabel)
                if i < 2:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Year")
                if i == 0:
                    ax.text(0.50, 0.96, scenario.upper().replace("SSP", "SSP "), transform=ax.transAxes, ha="center", va="top", fontsize=8.0)
                    panel_label_inside(ax, labels[col], x=0.012, y=0.96)

    annual_stack(
        2,
        annual_mode,
        ("annual_change_pct_mean", "annual_change_pct_p10", "annual_change_pct_p90"),
        "Annual transmission\nincrease rate (%)",
        ("g", "h"),
    )
    annual_stack(
        5,
        annual_temp,
        ("annual_temp_increase_c_mean", "annual_temp_increase_c_p10", "annual_temp_increase_c_p90"),
        "Annual mean temperature\nincrease (deg C)",
        ("i", "j"),
    )
    save_figure(fig, "main_figure_5_future_temperature_counterfactual_ssp", output_dir)
    plt.close(fig)


def plot_supplementary_figure_s1a(output_dir: Path = OUT_DIR) -> None:
    grid = pd.read_csv(DATA_DIR / "supplementary_figure_s1/factor_grid_values.csv")
    boundary = read_boundary("EPSG:4545")
    specs = [
        ("built_up_share_pct", "Built-up (%)", "YlOrBr", None),
        ("natural_green_share_pct", "Natural/green (%)", "YlGn", None),
        ("elevation_m", "Elevation (m)", "terrain", None),
        ("population_density_people_km2", "People km$^{-2}$", "magma", None),
        ("residential_share_pct", "Residential (%)", "PuRd", None),
        ("nearest_industrial_distance_km", "Industrial dist. (km)", "viridis_r", None),
        ("ndvi_p50", "NDVI p50", "Greens", None),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(183 / 25.4, 198 / 25.4))
    axes = axes.ravel()
    for i, (col, label, cmap, norm) in enumerate(specs):
        ax = axes[i]
        vals = pd.to_numeric(grid[col], errors="coerce")
        if norm is None:
            lo, hi = np.nanpercentile(vals, [2, 98])
            norm = Normalize(vmin=lo, vmax=hi)
        im = ax.scatter(grid["x"], grid["y"], c=vals, s=4.2, marker="s", cmap=cmap, norm=norm, linewidth=0, rasterized=True)
        boundary.boundary.plot(ax=ax, color="#2B3036", linewidth=0.55)
        setup_map_axis(ax, boundary)
        add_horizontal_colorbar(fig, ax, im, label, width="56%", y=-0.12)
        panel_label_inside(ax, chr(ord("a") + i))
    axes[-1].set_axis_off()
    save_figure(fig, "supplementary_figure_s1a_factor_maps", output_dir)
    plt.close(fig)


def plot_supplementary_figure_s1b(output_dir: Path = OUT_DIR) -> None:
    ctx = pd.read_csv(DATA_DIR / "supplementary_figure_s1/line_factor_values.csv")
    factors = [
        ("built_up_pct", "Built-up share (%)"),
        ("natural_green_pct", "Natural/green share (%)"),
        ("elevation_m", "Elevation (m)"),
        ("population_density", "Population density (people km$^{-2}$)"),
        ("residential_pct", "Residential share (%)"),
        ("industrial_distance_km", "Nearest industrial distance (km)"),
        ("ndvi_p50", "NDVI p50"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(183 / 25.4, 198 / 25.4), gridspec_kw={"hspace": 0.58, "wspace": 0.45})
    axes = axes.ravel()
    for i, (col, label) in enumerate(factors):
        ax = axes[i]
        data = ctx[["pattern", col]].dropna()
        sns.boxplot(
            data=data,
            x=col,
            y="pattern",
            hue="pattern",
            order=PATTERNS,
            hue_order=PATTERNS,
            ax=ax,
            palette=PATTERN_COLORS,
            width=0.48,
            fliersize=1.1,
            linewidth=0.6,
            legend=False,
        )
        sns.stripplot(data=data, x=col, y="pattern", order=PATTERNS, ax=ax, color="#111827", size=1.4, alpha=0.27, jitter=0.18)
        ax.set_ylabel("")
        ax.set_yticks(range(len(PATTERNS)))
        ax.set_yticklabels([pattern_short(p) for p in PATTERNS])
        ax.set_xlabel(label, labelpad=2)
        ax.grid(axis="x", color="#E5E7EB", lw=0.45)
        if col in {"population_density", "industrial_distance_km"} and data[col].min() > 0:
            ax.set_xscale("log")
        panel_label(ax, chr(ord("a") + i), x=-0.10, y=1.03)
    ax = axes[-1]
    med = ctx.groupby("pattern").agg(
        built_up=("built_up_pct", "median"),
        green=("natural_green_pct", "median"),
        elevation=("elevation_m", "median"),
        population=("population_density", "median"),
        cdd=("jas_cdd24_per_year", "median"),
        hdd=("ndjf_hdd18_per_year", "median"),
    )
    z = (med - med.mean()) / med.std(ddof=0)
    im = ax.imshow(z.loc[PATTERNS].T, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(3), [pattern_short(p) for p in PATTERNS])
    ax.set_yticks(range(len(z.columns)), ["Built", "Green", "Elev.", "Pop.", "JAS\nCDD24", "NDJF\nHDD18"])
    ax.tick_params(length=0)
    for r in range(z.shape[1]):
        for c in range(3):
            ax.text(c, r, f"{z.loc[PATTERNS[c], z.columns[r]]:.1f}", ha="center", va="center", fontsize=5.6)
    add_horizontal_colorbar(fig, ax, im, "standardized median", width="55%", y=-0.24)
    panel_label(ax, "h", x=-0.10, y=1.03)
    save_figure(fig, "supplementary_figure_s1b_factor_distributions", output_dir)
    plt.close(fig)


def plot_supplementary_figure_s2(output_dir: Path = OUT_DIR) -> None:
    compare = pd.read_csv(DATA_DIR / "supplementary_figure_s2/raw_vs_downscaled_access_cm2_line_summary.csv")
    annual = pd.read_csv(DATA_DIR / "supplementary_figure_s2/annual_temperature_ensemble.csv")
    lines = read_lines()
    boundary = read_boundary(lines.crs)
    map_df = compare[(compare["scenario"].eq("ssp585")) & (compare["year"].eq(2090))].copy()
    gdf = lines.merge(map_df, left_on="line_id", right_on="line_name", how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=lines.crs)
    fig = plt.figure(figsize=(183 / 25.4, 132 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.72, 0.84], hspace=0.28, wspace=0.26)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    temp_norm = Normalize(vmin=float(np.nanpercentile(gdf[["raw_annual_mean_c", "downscaled_annual_mean_c"]].to_numpy(), 3)), vmax=float(np.nanpercentile(gdf[["raw_annual_mean_c", "downscaled_annual_mean_c"]].to_numpy(), 97)))
    diff_abs = float(np.nanpercentile(np.abs(gdf["downscaled_minus_raw_c"]), 97))
    draw_value_line_map(fig, axes[0], gdf, "raw_annual_mean_c", cmap="YlOrRd", norm=temp_norm, label="deg C", boundary=boundary, linewidth=0.86)
    draw_value_line_map(fig, axes[1], gdf, "downscaled_annual_mean_c", cmap="YlOrRd", norm=temp_norm, label="deg C", boundary=boundary, linewidth=0.86)
    draw_value_line_map(fig, axes[2], gdf, "downscaled_minus_raw_c", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs), label="deg C", boundary=boundary, linewidth=0.86)
    for label, ax in zip("abc", axes):
        panel_label_inside(ax, label)

    ax_ts = fig.add_subplot(gs[1, :2])
    for scenario in ["ssp126", "ssp585"]:
        for pattern in PATTERNS:
            sub = annual[(annual["scenario"].eq(scenario)) & (annual["pattern"].eq(pattern))].sort_values("year")
            ls = {"Pattern 1": "-", "Pattern 2": "--", "Pattern 3": ":"}[pattern]
            ax_ts.plot(sub["year"], sub["annual_temp_increase_c_mean"], color=SCENARIO_COLORS[scenario], lw=1.0, ls=ls, label=f"{scenario.upper()} {pattern_short(pattern)}")
        scen = annual[annual["scenario"].eq(scenario)].groupby("year", as_index=False).agg(p10=("annual_temp_increase_c_p10", "mean"), p90=("annual_temp_increase_c_p90", "mean"))
        ax_ts.fill_between(scen["year"], scen["p10"], scen["p90"], color=SCENARIO_COLORS[scenario], alpha=0.11, lw=0)
    ax_ts.axhline(0, color="#6B7280", lw=0.5)
    ax_ts.set_ylabel("Annual temperature increase (deg C)")
    ax_ts.set_xlabel("Year")
    ax_ts.set_xlim(2026, 2100)
    ax_ts.grid(color="#E5E7EB", lw=0.45)
    ax_ts.legend(ncol=3, frameon=False, loc="upper left", fontsize=5.7)
    panel_label(ax_ts, "d")

    ax_box = fig.add_subplot(gs[1, 2])
    box = compare[(compare["year"].eq(2090)) & (compare["scenario"].eq("ssp585"))].copy()
    long = pd.concat(
        [
            box[["pattern", "raw_annual_mean_c"]].rename(columns={"raw_annual_mean_c": "temperature_c"}).assign(product="Raw"),
            box[["pattern", "downscaled_annual_mean_c"]].rename(columns={"downscaled_annual_mean_c": "temperature_c"}).assign(product="Downscaled"),
        ],
        ignore_index=True,
    )
    sns.boxplot(data=long, x="product", y="temperature_c", hue="pattern", hue_order=PATTERNS, palette=PATTERN_COLORS, ax=ax_box, fliersize=0.8, linewidth=0.6)
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Annual mean temperature (deg C)")
    ax_box.legend(title="", frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.52, 1.02), ncol=1)
    ax_box.grid(axis="y", color="#E5E7EB", lw=0.45)
    panel_label(ax_box, "e")
    save_figure(fig, "supplementary_figure_s2_future_temperature_downscaling", output_dir)
    plt.close(fig)


def raster_to_array(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], object, float | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return arr, extent, src.crs, nodata


def add_raster_map(ax: plt.Axes, path: Path, lines: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, vmin: float, vmax: float) -> mpl.image.AxesImage:
    arr, extent, crs, _ = raster_to_array(path)
    b = boundary.to_crs(crs)
    l = lines.to_crs(crs)
    cmap = mpl.colormaps["RdYlBu_r"].copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    b.boundary.plot(ax=ax, color="#20242A", linewidth=0.62, zorder=5)
    for pattern in PATTERNS:
        l[l["pattern_label"].eq(pattern)].plot(ax=ax, color=PATTERN_COLORS[pattern], linewidth=0.42, alpha=0.86, zorder=6)
    setup_map_axis(ax, b)
    return im


def plot_supplementary_figure_s3(output_dir: Path = OUT_DIR) -> None:
    line_year = pd.read_csv(DATA_DIR / "supplementary_figure_s3/line_year_temperature_degree_days.csv")
    lines = read_lines()
    boundary = read_boundary()
    rasters = [DATA_DIR / f"supplementary_figure_s3/rasters/hku_1km_wanzhou_annual_tmean_{year}.tif" for year in [2021, 2022, 2023, 2024]]
    all_vals = []
    for path in rasters:
        arr, *_ = raster_to_array(path)
        all_vals.append(arr[np.isfinite(arr)])
    vmin, vmax = np.nanpercentile(np.concatenate(all_vals), [2, 98])

    fig = plt.figure(figsize=(183 / 25.4, 158 / 25.4))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.74, 0.74, 0.10, 0.72], hspace=0.14, wspace=0.34)
    axes = [fig.add_subplot(gs[i // 2, (i % 2) * 2 : (i % 2) * 2 + 2]) for i in range(4)]
    ims = []
    for ax, year, path, label in zip(axes, [2021, 2022, 2023, 2024], rasters, "abcd"):
        ims.append(add_raster_map(ax, path, lines, boundary, float(vmin), float(vmax)))
        ax.text(0.94, 0.94, str(year), transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#111827")
        panel_label_inside(ax, label)
    cax = fig.add_subplot(gs[2, 1:3])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("Annual mean temperature (deg C)", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.35)

    ax_box = fig.add_subplot(gs[3, :2])
    sns.boxplot(data=line_year, x="year", y="annual_tmean_c", hue="pattern", hue_order=PATTERNS, palette=PATTERN_COLORS, ax=ax_box, linewidth=0.6, fliersize=0.8)
    ax_box.set_xlabel("Year")
    ax_box.set_ylabel("Line annual mean (deg C)")
    ax_box.legend(title="", ncol=3, frameon=False, fontsize=5.8)
    ax_box.grid(axis="y", color="#E5E7EB", lw=0.45)
    panel_label(ax_box, "e")

    ax_jas = fig.add_subplot(gs[3, 2:])
    med = line_year.groupby(["year", "pattern"], as_index=False).agg(jas=("jas_tmean_c", "median"))
    for pattern in PATTERNS:
        sub = med[med["pattern"].eq(pattern)]
        ax_jas.plot(sub["year"], sub["jas"], marker="o", ms=3, lw=1.1, color=PATTERN_COLORS[pattern], label=pattern_short(pattern))
    ax_jas.set_xlabel("Year")
    ax_jas.set_ylabel("Median JAS (deg C)", labelpad=2)
    ax_jas.set_xticks([2021, 2022, 2023, 2024])
    ax_jas.grid(color="#E5E7EB", lw=0.45)
    ax_jas.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax_jas, "f")
    save_figure(fig, "supplementary_figure_s3_hku_1km_multiyear_temperature", output_dir)
    plt.close(fig)


def plot_supplementary_figure_s4(output_dir: Path = OUT_DIR) -> None:
    line_year = pd.read_csv(DATA_DIR / "supplementary_figure_s4/line_year_temperature_degree_days.csv")
    month = pd.read_csv(DATA_DIR / "supplementary_figure_s4/pattern_month_degree_days.csv")
    lines = read_lines()
    boundary = read_boundary(lines.crs)
    mean_line = line_year.groupby(["line_id", "pattern"], as_index=False).agg(mean_cdd24=("cdd24", "mean"), mean_hdd18=("hdd18", "mean"))
    gdf = lines.merge(mean_line, left_on=["line_id", "pattern_label"], right_on=["line_id", "pattern"], how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=lines.crs)
    fig = plt.figure(figsize=(183 / 25.4, 135 / 25.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.74, 0.84], hspace=0.30, wspace=0.35)
    ax_cdd = fig.add_subplot(gs[0, :2])
    ax_hdd = fig.add_subplot(gs[0, 2:])
    draw_value_line_map(fig, ax_cdd, gdf, "mean_cdd24", cmap="YlOrRd", norm=Normalize(vmin=0, vmax=np.nanpercentile(gdf["mean_cdd24"], 98)), label="deg C day yr$^{-1}$", boundary=boundary, linewidth=0.82)
    draw_value_line_map(fig, ax_hdd, gdf, "mean_hdd18", cmap="PuBu", norm=Normalize(vmin=np.nanpercentile(gdf["mean_hdd18"], 2), vmax=np.nanpercentile(gdf["mean_hdd18"], 98)), label="deg C day yr$^{-1}$", boundary=boundary, linewidth=0.82)
    panel_label_inside(ax_cdd, "a")
    panel_label_inside(ax_hdd, "b")

    ax_box = fig.add_subplot(gs[1, :2])
    long = pd.concat(
        [
            line_year[["pattern", "cdd24"]].rename(columns={"cdd24": "degree_days"}).assign(metric="CDD24"),
            line_year[["pattern", "hdd18"]].rename(columns={"hdd18": "degree_days"}).assign(metric="HDD18"),
        ],
        ignore_index=True,
    )
    sns.boxplot(data=long, x="metric", y="degree_days", hue="pattern", hue_order=PATTERNS, palette=PATTERN_COLORS, ax=ax_box, linewidth=0.6, fliersize=0.8)
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Line degree-day exposure (deg C day yr$^{-1}$)")
    ax_box.legend(title="", frameon=False, ncol=3, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax_box.grid(axis="y", color="#E5E7EB", lw=0.45)
    panel_label(ax_box, "c")

    ax_heat = fig.add_subplot(gs[1, 2:])
    cdd = month.pivot(index="pattern", columns="month", values="median_cdd24").loc[PATTERNS]
    hdd = month.pivot(index="pattern", columns="month", values="median_hdd18").loc[PATTERNS]
    balance = cdd - hdd
    vmax = float(np.nanmax(np.abs(balance.to_numpy())))
    im = ax_heat.imshow(balance, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_yticks(range(3), [pattern_short(p) for p in PATTERNS])
    ax_heat.set_xticks(range(12), list(range(1, 13)))
    ax_heat.set_xlabel("Month")
    ax_heat.tick_params(length=0)
    add_horizontal_colorbar(fig, ax_heat, im, "CDD24 minus HDD18 (deg C day month$^{-1}$)", width="68%", y=-0.25)
    panel_label(ax_heat, "d")
    save_figure(fig, "supplementary_figure_s4_hku_1km_degree_day_context", output_dir)
    plt.close(fig)


FIGURE_FUNCTIONS = {
    "main1": plot_main_figure_1,
    "main2": plot_main_figure_2,
    "main3": plot_main_figure_3,
    "main4": plot_main_figure_4,
    "main5": plot_main_figure_5,
    "s1a": plot_supplementary_figure_s1a,
    "s1b": plot_supplementary_figure_s1b,
    "s2": plot_supplementary_figure_s2,
    "s3": plot_supplementary_figure_s3,
    "s4": plot_supplementary_figure_s4,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce all main and supplementary manuscript figures from the packaged minimal data.")
    parser.add_argument("--figure", choices=[*FIGURE_FUNCTIONS.keys(), "all"], default="all")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    configure_matplotlib()
    output_dir = Path(args.output_dir)
    selected = FIGURE_FUNCTIONS.keys() if args.figure == "all" else [args.figure]
    for key in selected:
        print(f"Rendering {key}...")
        FIGURE_FUNCTIONS[key](output_dir)
    cleanup_appledouble(PKG_DIR)
    print(f"Done. Figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
