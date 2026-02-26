# Rethinking Metrics for Lexical Semantic Change Detection

This repo explores **lexical semantic change detection (LSCD)** metrics when word meaning is represented in different vector spaces:

- **Full space**: contextual usage embeddings as-is
- **Definition space**: project usage embeddings onto a basis induced by **LLM-generated dictionary definitions**
- **PCA space**: reduce the joint usage space to a controlled dimensionality
- **Random subspace**: keep a random subset of dimensions as a sanity check / control

The main evaluation is: for each language and encoder, compute LSCD metrics between time periods and measure **Spearman correlation** against SemEval-style gold graded change scores.

The repository is notebook-driven, with a few core Python modules providing reusable functionality.

## What’s in here

- Load SemEval / DWUG-style LSCD datasets across multiple languages.
- Generate dictionary-style definitions for each lemma (via external LLM APIs).
- Embed definitions and contextual usages with HuggingFace encoders.
- Compare LSCD metrics (e.g. **APD**, **PRT**, **AMD**, and **SAMD**) across spaces.
- Aggregate results into CSVs and visualize them (heatmaps, hubness analysis).

## Repository layout

- [def_proj_functions.py](def_proj_functions.py): dataset loading, encoder loading, span-based embedding utilities, correlation helpers.
- [embed_defs_functions.py](embed_defs_functions.py): load definition JSON, embed definitions, correlation helper used by notebooks.
- [spaces.py](spaces.py): space transforms (full/definition/PCA/random).
- [metrics.py](metrics.py): LSCD metric implementations (APD/PRT/AMD + symmetric matching variants).
- [vis_functions.py](vis_functions.py): result aggregation + heatmap plotting.
- Notebooks:
	- [generate_defs.ipynb](generate_defs.ipynb): generate definitions (requires API keys).
	- [embed_defs.ipynb](embed_defs.ipynb): embed definition sets.
	- [different_spaces_run_exp.ipynb](different_spaces_run_exp.ipynb): run the main experiments and write results.
	- [data_analysis.ipynb](data_analysis.ipynb): aggregate/analyse results.
	- [hubness.ipynb](hubness.ipynb), [hubness_visualisation.ipynb](hubness_visualisation.ipynb): hubness diagnostics.
- Results:
	- [results/master_results_spaces.csv](results/master_results_spaces.csv): long-form results table (language × encoder × metric × space).
	- [results/hubness.csv](results/hubness.csv): hubness statistics.

## Installation

This project isn’t packaged as a library; you typically run it from a clone.
Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The code expects your LSCD datasets to be available locally.
- Set `SEMEVAL_DATA_ROOT` to the directory that contains the dataset folders referenced in [def_proj_functions.py](def_proj_functions.py).

## Usage

Most workflows are executed through notebooks.

### 1) Prepare datasets

Ensure `SEMEVAL_DATA_ROOT` points to a directory containing (at least some of):

- `semeval2020_ulscd_eng/`
- `semeval2020_ulscd_ger/`
- `semeval2020_ulscd_swe/`
- `semeval2020_ulscd_lat/`
- `dwug_es/`
- `chiwug/`
- `nor_dia_change/`

Then you can load a language dataset from Python:

```python
from def_proj_functions import load_semeval_df

df = load_semeval_df("english")
df.head()
```

### 2) (Optional) Generate definitions

If you want to regenerate definitions, run:

- [generate_defs.ipynb](generate_defs.ipynb)

Definition text is stored as JSON mapping `lemma -> [definition, ...]` under `definitions/`.

### 3) Embed definitions

Run:

- [embed_defs.ipynb](embed_defs.ipynb)

The helper in [embed_defs_functions.py](embed_defs_functions.py) uses span-aware token alignment via `get_positions()` and `get_wordtransformer_embeddings()`.

### 4) Run experiments (spaces + metrics)

Run:

- [different_spaces_run_exp.ipynb](different_spaces_run_exp.ipynb)

This evaluates metrics from [metrics.py](metrics.py) across space transforms defined in [spaces.py](spaces.py), and writes/updates long-form results in:

- [results/master_results_spaces.csv](results/master_results_spaces.csv)

### 5) Analyse + visualise

For aggregation and plots, see:

- [data_analysis.ipynb](data_analysis.ipynb)
- [hubness_visualisation.ipynb](hubness_visualisation.ipynb)

The main plotting utilities live in [vis_functions.py](vis_functions.py) (notably `plot_effect_heatmap`).

## Example outputs

### Long-form results table

The file [results/master_results_spaces.csv](results/master_results_spaces.csv) stores one row per `(language, encoder, definition model, metric, space)` with a Spearman correlation.

### Hubness diagnostics

The file [results/hubness.csv](results/hubness.csv) contains hubness statistics about the four embedding space types we analyse.


