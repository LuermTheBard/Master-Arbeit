# `plot_1D_ccfs_and_reference_lightcurves.py` — Documentation

This module generates publication-quality figures showing **1D cross-correlation functions (CCFs)** alongside their associated **reference lightcurves** for the NGC 4593 reverberation mapping campaign. Each figure consists of a side-by-side subplot layout: normalized lightcurve pairs on the left, and the corresponding CCF on the right.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Module Overview](#2-module-overview)
3. [PlotConfig — Layout & Appearance Settings](#3-plotconfig--layout--appearance-settings)
4. [Predefined Configurations](#4-predefined-configurations)
5. [Function Reference](#5-function-reference)
   - [save_1d_corr_and_lightcurves_general](#save_1d_corr_and_lightcurves_general)
   - [plot_1d_corr_and_lightcurves_in_groups](#plot_1d_corr_and_lightcurves_in_groups)
   - [plot_ccfs_and_reference_lightcurves_in_groups](#plot_ccfs_and_reference_lightcurves_in_groups)
   - [prepare_ccfs_references_data](#prepare_ccfs_references_data)
   - [configure_ccfs_and_reference_axis](#configure_ccfs_and_reference_axis)
   - [configure_axes_for_lightcurves](#configure_axes_for_lightcurves)
   - [configure_axes_for_ccfs](#configure_axes_for_ccfs)
   - [check_for_empty_rows_ccfs_and_reference](#check_for_empty_rows_ccfs_and_reference)
   - [finalize_figure_ccfs_and_reference](#finalize_figure_ccfs_and_reference)
   - [calculate_standard_error_for_lightcurves](#calculate_standard_error_for_lightcurves)
   - [normalize_lightcurve](#normalize_lightcurve)
   - [deep_merge](#deep_merge)
6. [Helper Functions](#6-helper-functions)
7. [Predefined Figure Functions](#7-predefined-figure-functions)
8. [Command-Line Interface](#8-command-line-interface)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Dependencies](#10-dependencies)

---

## 1. Quick Start

```bash
# List all available figures
python plot_1D_ccfs_and_reference_lightcurves.py --list

# Generate and save a single figure
python plot_1D_ccfs_and_reference_lightcurves.py fig1

# Generate and display a figure interactively
python plot_1D_ccfs_and_reference_lightcurves.py fig1 --show

# Generate multiple figures at once
python plot_1D_ccfs_and_reference_lightcurves.py fig1 fig2 fig3

# Generate all figures
python plot_1D_ccfs_and_reference_lightcurves.py all
```

Output files (PDF + PNG) are saved to `default_output/corr_and_lightcurves/`.

---

## 2. Module Overview

The module is organized into five logical sections:

| Section | Purpose |
|---|---|
| **PlotConfig** | Dataclass controlling all layout/appearance options |
| **Data Import & Preparation** | Loads CCF, lightcurve, and centroid data; merges campaigns |
| **Data Processing / Sorting** | Sorts and groups data by reference lightcurve and line order |
| **Plot Creation** | Creates the subplot figures and configures individual axes |
| **Figure Functions** | High-level wrappers for each named publication figure |

---

## 3. `PlotConfig` — Layout & Appearance Settings

`PlotConfig` is a `dataclass` that centralises all layout and appearance options. Pass an instance (or a modified copy via `dataclasses.replace()`) to any plotting function.

```python
from dataclasses import replace
config = replace(PAPER_CONFIG, rows=6, figsize=(6, 8))
```

| Field | Type | Default | Description |
|---|---|---|---|
| `rows` | `int` | `8` | Number of subplot rows per figure |
| `cols` | `int` | `2` | Number of columns (left = lightcurve, right = CCF) |
| `figsize` | `tuple \| None` | `None` | Figure size in inches `(width, height)`. `None` = auto-calculated |
| `combine_data` | `bool` | `True` | Merge optically and non-optically calibrated datasets |
| `show_reference_label` | `bool` | `False` | Show the reference lightcurve name in the legend |
| `format_labels_as_paper` | `bool` | `False` | Use compact LaTeX labels (for publications) |
| `layout_show_right_ccf_ylabel` | `bool` | `True` | Show y-axis label on the right side of CCF panels |
| `layout_show_top_secondary_labels` | `bool` | `True` | Show date labels on the secondary top x-axis |
| `lightcurve_hide_yticklabels` | `bool` | `True` | Hide y-axis tick labels on lightcurve panels |
| `ccf_show_inline_label_text` | `bool` | `True` | Show the line name as inline text inside the CCF panel |
| `adjust_last_row_gap_inch` | `float` | `0.0` | Vertical offset of the last row in inches (negative = tighter) |
| `include_extra_data` | `bool` | `False` | Append an additional dataset as the last row |
| `extra_data_name` | `str \| None` | `None` | Key of the extra dataset, format: `"<line>_ref_<reference>"` |
| `show_histogram` | `bool \| None` | `None` | Show MCMC centroid histogram inside the CCF panel |
| `show_subfigure_labels` | `bool` | `True` | Show sub-figure labels `a)`, `b)`, … |
| `row_spacing` | `float \| None` | `None` | Vertical spacing between rows (`hspace`). `None` = none |
| `line_style` | `str` | `"-"` | Matplotlib line style, e.g. `"-"` or `":"` |
| `grid` | `tuple \| None` | `None` | Grid settings `(show_minor, alpha, linewidth, linestyle)` or `None` |

---

## 4. Predefined Configurations

Two ready-to-use configurations are provided:

### `EXPLORE_CONFIG`
All layout helpers enabled; intended for interactive exploration.
```python
EXPLORE_CONFIG = PlotConfig()
```

### `PAPER_CONFIG`
Clean, minimal layout intended for publications and theses.
```python
PAPER_CONFIG = PlotConfig(
    show_reference_label=True,
    format_labels_as_paper=True,
    layout_show_right_ccf_ylabel=False,
    layout_show_top_secondary_labels=False,
    lightcurve_hide_yticklabels=False,
    ccf_show_inline_label_text=False,
    adjust_last_row_gap_inch=-0.2,
    show_histogram=False,
    show_subfigure_labels=False,
    line_style="-",
    grid=(True, 0.12, 0.3, ':'),
)
```

---

## 5. Function Reference

---

### `save_1d_corr_and_lightcurves_general`

**Top-level entry point** for generating a complete figure. Imports all data, optionally merges campaigns, and delegates to `plot_1d_corr_and_lightcurves_in_groups`.

```python
save_1d_corr_and_lightcurves_general(
    campaign_keys,
    keyorders_dict,
    output_dir=DEFAULT_OUTPUT_DIR,
    file_name="ccfs_and_reference_lightcurves",
    final_key_order=None,
    campaign_label=None,
    config: PlotConfig = None,
    save_only: bool = True,
)
```

| Parameter | Type | Description |
|---|---|---|
| `campaign_keys` | `list[str]` | Campaign identifiers to process. Ignored when `config.combine_data=True` |
| `keyorders_dict` | `dict` | Maps each reference lightcurve to an ordered list of lines to plot. Format: `{"<reference>": ["time shift (tau)", "<line1>", ...]}` |
| `output_dir` | `Path` | Root output directory |
| `file_name` | `str` | Base name for the output files |
| `final_key_order` | `list \| None` | Final sort order for the plotted lines across all references |
| `campaign_label` | `str \| None` | Label used for the combined campaign (default: `"Combined"`) |
| `config` | `PlotConfig \| None` | Layout settings (default: `PlotConfig()`) |
| `save_only` | `bool` | If `True`, figures are saved to disk without displaying |

**Behaviour with `combine_data=True`:** Merges `NGC4593_optical_calibrated` and `NGC4593_not_optical_calibrated` using `deep_merge` before plotting.

---

### `plot_1d_corr_and_lightcurves_in_groups`

**Orchestrates data sorting and grouping** before passing to the actual plot creation. Iterates over each reference lightcurve in `key_orders`, extracts and renames the relevant CCF and lightcurve data, resolves the `extra_data_name` if set, and calls `plot_ccfs_and_reference_lightcurves_in_groups`.

```python
plot_1d_corr_and_lightcurves_in_groups(
    lightcurves_ccf_data_dict,
    campaign,
    output_dir,
    key_orders,
    save_only=False,
    file_name=None,
    final_key_order=None,
    only_one_label=False,
    centroid_data=None,
    config: PlotConfig = None,
)
```

**Key logic:**
- Keys are renamed to `"<line>_ref_<reference>"` to encode both the line and its reference in the dictionary key.
- A second sorting pass using `final_key_order` ensures a consistent row order across references.
- If `config.include_extra_data` is set, the extra entry is appended at the end of the sorted dict.

---

### `plot_ccfs_and_reference_lightcurves_in_groups`

**Creates and populates the Matplotlib figure.** Iterates over groups of data produced by `prepare_ccfs_references_data` and calls `configure_ccfs_and_reference_axis` for each subplot cell. After all axes are filled, calls `check_for_empty_rows_ccfs_and_reference` and `finalize_figure_ccfs_and_reference`.

```python
plot_ccfs_and_reference_lightcurves_in_groups(
    final_sorted_data_dict,
    xlabel_ccfs,
    ylabel_ccfs,
    xlabel_lightcurves,
    save_only,
    output_dir,
    shared_y,
    file_name,
    centroid_data=None,
    only_one_label=False,
    config: PlotConfig = None,
    panel_height=1.2,       # inches per row
    right_panel_width=1.2,  # inches for the CCF column
    padding=(1.0, 1.6),     # (width_pad, height_pad) in inches
)
```

**Layout details:**
- Uses a `gridspec_kw={'width_ratios': [4, 1]}` grid so the lightcurve panel is 4× wider than the CCF panel.
- All axes in the same column share their x-axis.
- Even-indexed subplots (col 0) receive lightcurve data; odd-indexed (col 1) receive CCF data.

---

### `prepare_ccfs_references_data`

**Generator** that duplicates each entry (once for lightcurve, once for CCF) and yields chunks of `rows × cols` items padded with `None` placeholders.

```python
for current_data, group_index in prepare_ccfs_references_data(data, rows, cols):
    ...
```

| Parameter | Description |
|---|---|
| `data` | Dict of `"<line>_ref_<reference>"` → data entries |
| `rows` | Rows per subplot group |
| `cols` | Columns per subplot group |

**Yields:** `(current_data, group_index)` where `current_data` is a list of `(key, value)` tuples.

---

### `configure_ccfs_and_reference_axis`

**Configures a single subplot axis**, delegating to either the lightcurve or CCF branch based on the column index.

```python
configure_ccfs_and_reference_axis(
    ax, row, rows, col,
    ylabel_ccfs, color,
    x_values_ccfs, line_data, yerr,
    line_name_and_ref_name,
    centroid_data=None,
    only_one_label=False,
    config: PlotConfig = None,
)
```

**Lightcurve branch (col = 0):**
- Normalizes both the line and reference lightcurves using `normalize_lightcurve`.
- Retrieves marker style and color from `SYMBOLES_AND_COLORS_FOR_LIGHTCURVES`.
- Applies optional error correction (`ERR_CORRECTION`, `ERR_SET` from `settings.py`).
- Plots both lightcurves as `errorbar` with caps; adds sub-figure label if `config.show_subfigure_labels`.
- Calls `configure_axes_for_lightcurves`.

**CCF branch (col = 1):**
- Plots the CCF curve as a line.
- Loads MC centroid/peak data via `import_centroid_and_mc_data`.
- If `centroid_data` is provided, annotates the centroid lag τ with asymmetric error bars and draws a vertical dashed line.
- Optionally overlays the MCMC centroid histogram (`config.show_histogram`).
- Adds inline line label if `config.ccf_show_inline_label_text`.
- Calls `configure_axes_for_ccfs`.

---

### `configure_axes_for_lightcurves`

Applies standard axis formatting to a **lightcurve panel**:
- Y-limits fixed to `[-3, 3]` (normalized units).
- Major tick every 5 MJD; minor tick every 1 MJD.
- Secondary top x-axis with date labels (`'Mon DD'`) on the first row if `layout_show_top_secondary_labels=True`.
- Hides y-tick labels when `lightcurve_hide_yticklabels=True`.

---

### `configure_axes_for_ccfs`

Applies standard axis formatting to a **CCF panel**:
- X-limits `[-5, 10]` days; Y-limits `[0, 1]`.
- Y-axis on the right side; major ticks at 0.5, minor at 0.1.
- Zero-lag reference line (dotted, `linewidth=0.5`).
- Smart tick label suppression: `"0"` is shown only on the bottom and second-to-last rows; `"1"` is shown on every row.

---

### `check_for_empty_rows_ccfs_and_reference`

Post-processing step that:
1. Removes axes belonging to entirely empty rows from the figure.
2. Ensures tick labels are shown only on the lowest visible row.
3. Attaches the x-axis labels to the lowest visible row.
4. Optionally shifts the last row upward by `adjust_last_row_gap_inch` to tighten the layout.

---

### `finalize_figure_ccfs_and_reference`

Saves the figure as both **PDF** and **PNG** to `output_dir`, prints the save path, and optionally calls `plt.show()` before closing.

```python
finalize_figure_ccfs_and_reference(fig, filename, save_only, output_dir)
```

---

### `calculate_standard_error_for_lightcurves`

Computes the total photometric uncertainty:

```
total_error = sqrt((F_VAR × flux)² + flux_noise_err²)
```

Optional modifiers:
- `err_correction` (float): adds a relative percentage correction on top (`total_error *= 1 + err_correction/100`).
- `err_set` (float): overrides the error entirely with `flux × err_set / 100`.

`F_VAR` is imported from `settings.py`.

---

### `normalize_lightcurve`

Z-score normalizes flux values and propagates uncertainties:

```python
y_norm   = (y - mean(y)) / std(y)
yerr_norm = total_error / std(y)
```

Calls `calculate_standard_error_for_lightcurves` internally.

---

### `deep_merge`

Recursively merges two nested dictionaries. Values in `dict2` take precedence; nested dicts are merged rather than replaced.

```python
merged = deep_merge(dict1, dict2)
```

---

## 6. Helper Functions

| Function | Description |
|---|---|
| `mjd_to_date(mjd)` | Converts a Modified Julian Date to a `datetime.datetime` object (epoch: 1858-11-17) |
| `format_month_day(mjd, pos)` | Matplotlib tick formatter; converts MJD → `'Mon DD'` string (e.g. `'Aug 01'`) |
| `is_valid_axis(ax, fig)` | Returns `True` if `ax` is part of `fig` and contains at least one plotted element |
| `_apply_grid(ax, grid)` | Draws a major (and optionally minor) grid on `ax` using the `grid` tuple from `PlotConfig` |

---

## 7. Predefined Figure Functions

Each function wraps `save_1d_corr_and_lightcurves_general` with a fixed `keyorders_dict` and `PlotConfig`. All accept `save_only: bool = True`.

| Function | CLI key | Reference | Lines | Output file |
|---|---|---|---|---|
| `figure_1_uvw2_balmer_ly_o` | `fig1` | UVW2 | Hα, Hβ, Hγ, Hδ, Lyα, OI 8446 | `UVW2_ccfs_Balmer_Ly_O` |
| `figure_2_uvw2_helium_uv` | `fig2` | UVW2 | He I 5876, He II 1640, He II 4686, N V, Si IV, C IV | `UVW2_ccfs_Helium_UV` |
| `figure_3_oi_second_paper_halpha` | `fig4` | Lyα, Hα, UVW2 | OI 8446 | `OI_ccfs_and_reference_lightcurves_second_paper` |
| `figure_4_oi_paper_halpha` | `fig3` | Lyα, UVW2 | OI 8446, Hα | `OI_ccfs_and_reference_lightcurves_paper_HAlpha` |
| `figure_5_oi_hst_uv_halpha` | `fig5` | Lyα, Cont1150 | OI 8446, Hα | `OI_ccfs_and_reference_lightcurves_HST_UV_paper_HAlpha` |

> **Note:** fig1 and fig2 append `"UVW2_ref_Cont1150_not_optical_calibrated"` as an extra row via `include_extra_data`.

---

## 8. Command-Line Interface

```
usage: plot_1D_ccfs_and_reference_lightcurves.py [-h] [--show] [--list] [FIGURE ...]

positional arguments:
  FIGURE         fig1 … fig5, or 'all'

optional arguments:
  --show, -s     Display figures after saving
  --list, -l     Print available figure names and exit
```

The `FIGURES` registry maps CLI names to `(function, description)` pairs and is easily extended by adding a new entry.

---

## 9. Data Flow Diagram

```
import_1d_correlation_data()   ──┐
import_1d_lightcurve_data()    ──┤
load_centroid_data_by_reference() ─┤
                                   ▼
              save_1d_corr_and_lightcurves_general()
                      │  deep_merge (if combine_data)
                      ▼
       plot_1d_corr_and_lightcurves_in_groups()
            │  sort by key_orders / final_key_order
            │  rename keys → "<line>_ref_<reference>"
            │  attach extra_data if include_extra_data
            ▼
  plot_ccfs_and_reference_lightcurves_in_groups()
       │  prepare_ccfs_references_data() → (chunk, idx)
       │  plt.subplots() with width_ratios=[4,1]
       │  for each cell:
       │      configure_ccfs_and_reference_axis()
       │          col 0 → normalize_lightcurve() → errorbar()
       │                  configure_axes_for_lightcurves()
       │          col 1 → plot CCF + centroid annotation
       │                  configure_axes_for_ccfs()
       │  check_for_empty_rows_ccfs_and_reference()
       ▼
  finalize_figure_ccfs_and_reference()
       → save as .pdf + .png
```

---

## 10. Dependencies

| Package | Usage |
|---|---|
| `matplotlib` | Figure and axis creation, tick formatting |
| `numpy` | Array operations, normalization, error propagation |
| `pathlib` | Path handling |
| `dataclasses` | `PlotConfig` definition and `replace()` copies |
| `import_data` | `import_1d_correlation_data`, `import_1d_lightcurve_data`, `load_centroid_data_by_reference`, `import_centroid_and_mc_data` |
| `plot_utils` | `format_label`, `ensure_output_dir` |
| `settings` | `SYMBOLES_AND_COLORS_FOR_LIGHTCURVES`, `NUMBER_MAPPING`, `ERR_CORRECTION`, `ERR_SET`, `F_MEAN`, `F_VAR` |
