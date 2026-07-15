# Wanzhou manuscript figure reproduction package

This package contains the minimal derived data and plotting code needed to regenerate the PNG versions of the main-manuscript and supplementary figures.

It does not include raw CMIP/NEX-GDDP files, raw daily HKU temperature files, model training panels, or original GIS source layers. The packaged files are figure-level or figure-ready derivatives.

## Quick start

From this directory:

```bash
python run_all_figures.py
```

The command writes 10 PNG files to `output/`.

To render one figure:

```bash
python scripts/plot_figures.py --figure main5
python scripts/plot_figures.py --figure s4
```

Available figure keys are `main1`, `main2`, `main3`, `main4`, `main5`, `s1a`, `s1b`, `s2`, `s3`, and `s4`.

## Outputs

- `main_figure_1_operating_modes.png`
- `main_figure_2_temperature_response_bivariate_line_maps.png`
- `main_figure_3_rf_model_performance.png`
- `main_figure_4_rf_shap_pattern_mechanism.png`
- `main_figure_5_future_temperature_counterfactual_ssp.png`
- `supplementary_figure_s1a_factor_maps.png`
- `supplementary_figure_s1b_factor_distributions.png`
- `supplementary_figure_s2_future_temperature_downscaling.png`
- `supplementary_figure_s3_hku_1km_multiyear_temperature.png`
- `supplementary_figure_s4_hku_1km_degree_day_context.png`

Only PNG outputs are generated.

## Data layout

- `data/common/` contains the trimmed line geometries and Wanzhou district boundary used by map panels.
- `data/main_figure_*/` contains the figure-ready tables for the five main figures.
- `data/supplementary_figure_s*/` contains the figure-ready tables or clipped annual GeoTIFFs for the supplementary figures.
- `data/data_manifest.csv` records the package file, supported figure, description, original project source path, and file size.

For Main Figure 1, the daily heatmap is reproduced from two lossless display derivatives. `line_daily_heatmap_colors.png` stores exactly one 8-bit RGB color per heatmap cell using the published `RdBu_r` scale (0.60-1.72), while `line_daily_pattern_strip.png` stores only the anonymized row-level operating-pattern colors. Neither file contains line identifiers, raw transmission values, or continuous line-day normalized values. The monthly summary ribbons remain based on aggregated monthly profiles. For Supplementary Figure S3, the GeoTIFFs are already clipped annual HKU 1 km Wanzhou mean-temperature rasters, not raw daily temperature rasters.
