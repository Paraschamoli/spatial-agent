# Cell-Cell Communication Analysis

Analyze cell-cell interactions and communication dynamics in spatial transcriptomics data.

## Prerequisites

**Required**: Data must have cell type annotations in `obs`.
- If data is not annotated, run the **annotation** skill first

**Optional**: Spatial niche annotations for region-specific analysis.

---

## Workflow Overview

1. **Explore dataset structure** (always do this first)
2. Characterize dataset (cell types, conditions)
3. Infer cell-cell interactions (LIANA + CellPhoneDB + Tensor)
4. Analyze spatial context (if niches annotated)
5. Compare across conditions (if applicable)
6. Generate summary report

---

## Step 1: Explore Dataset Structure

**ALWAYS start here.** Use `execute_python` to understand the data:

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/data.h5ad")

# Basic info
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")

# Check for required columns
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    print(f"  {col}: {n_unique} unique values")
    if n_unique < 20:
        print(f"    values: {adata.obs[col].unique().tolist()}")
```

**Identify**:
- Cell type column name (e.g., `cell_type`, `celltype`, `annotation`)
- Sample/batch column name (e.g., `sample`, `batch`, `sample_id`)
- Condition column name (e.g., `condition`, `group`, `treatment`)

---

## Step 2: Characterize Dataset

```python
# Summarize cell type composition
result = summarize_celltypes({
    "adata_path": "path/to/data.h5ad",
    "celltype_key": "cell_type",  # Adjust based on your data
    "save_path": SAVE_PATH
})
print(result)

# Summarize conditions if available
if "condition" in adata.obs.columns:
    result = summarize_conditions({
        "adata_path": "path/to/data.h5ad",
        "condition_key": "condition",
        "save_path": SAVE_PATH
    })
    print(result)
```

---

## Step 3: LIANA Cell-Cell Communication

```python
# Run LIANA inference for cell-cell communication
result = liana_inference({
    "adata_path": "path/to/data.h5ad",
    "save_path": SAVE_PATH
})
print(result)

# Run LIANA spatial analysis
result = liana_spatial({
    "adata_path": "path/to/data.h5ad",
    "save_path": SAVE_PATH
})
print(result)
```

This identifies:
- Ligand-receptor interactions
- Interaction strengths
- Spatially restricted interactions

---

## Step 4: CellPhoneDB Analysis

```python
# Prepare data for CellPhoneDB
result = cellphonedb_prepare({
    "adata_path": "path/to/data.h5ad",
    "meta_path": SAVE_PATH,  # Will create metadata file
    "save_path": SAVE_PATH
})
print(result)

# Run CellPhoneDB analysis
result = cellphonedb_analysis({
    "adata_path": "path/to/data.h5ad",
    "meta_path": os.path.join(SAVE_PATH, "cellphonedb_meta.txt"),
    "save_path": SAVE_PATH
})
print(result)

# Filter and visualize results
result = cellphonedb_filter({
    "results_path": os.path.join(SAVE_PATH, "cellphonedb_results.txt"),
    "threshold": 0.05,
    "save_path": SAVE_PATH
})
print(result)

result = cellphonedb_plot({
    "results_path": os.path.join(SAVE_PATH, "filtered_results.txt"),
    "save_path": SAVE_PATH
})
print(result)
```

---

## Step 5: Spatial Context Analysis

If spatial niches are available:

```python
# Analyze communication within tissue niches
result = summarize_tissue_regions({
    "adata_path": "path/to/data.h5ad",
    "region_key": "tissue_niche",
    "save_path": SAVE_PATH
})
print(result)
```

---

## Step 6: Tensor Decomposition (Advanced)

```python
# Run LIANA tensor decomposition for higher-order interactions
result = liana_tensor({
    "adata_path": "path/to/data.h5ad",
    "save_path": SAVE_PATH
})
print(result)

# Run MISTY analysis for spatial patterns
result = liana_misty({
    "adata_path": "path/to/data.h5ad",
    "save_path": SAVE_PATH
})
print(result)
```

---

## Step 7: Generate Report

```python
# Generate comprehensive communication analysis report
result = report_subagent({
    "task_description": "Cell-cell communication analysis using LIANA and CellPhoneDB",
    "data_context": f"Dataset with {adata.n_obs} cells and {len(adata.obs['cell_type'].unique())} cell types",
    "save_path": SAVE_PATH,
    "report_type": "detailed"
})
print(result)
```

---

## Expected Outputs

1. **LIANA results**: Ligand-receptor interactions with scores
2. **CellPhoneDB results**: Significant interactions and p-values
3. **Spatial interaction maps**: Visualizations of communication patterns
4. **Tensor decomposition**: Higher-order interaction patterns
5. **Comprehensive report**: Summary of all findings

---

## Common Issues

**No cell type annotations**:
- Run the annotation skill first
- Or provide manual cell type labels

**Memory issues with CellPhoneDB**:
- Reduce dataset size by filtering
- Use subset of cell types
- Increase available memory

**No spatial coordinates**:
- LIANA can work without spatial data
- CellPhoneDB works with non-spatial data
- Spatial analysis requires coordinates

---

## Advanced Options

**Custom ligand-receptor databases**:
- Provide custom LR pairs
- Use tissue-specific databases
- Filter by expression thresholds

**Comparative analysis**:
```python
# Compare communication between conditions
if "condition" in adata.obs.columns:
    conditions = adata.obs["condition"].unique()
    for condition in conditions:
        subset = adata[adata.obs["condition"] == condition]
        # Run communication analysis on subset
```

**Network analysis**:
- Build communication networks
- Identify hub interactions
- Network topology analysis

---

## Quality Control

Always check:
- **Cell type annotation quality**
- **Expression of ligands/receptors**
- **Statistical significance** of interactions
- **Biological plausibility** of findings

<conclude>
Cell-cell communication analysis completed. Your analysis includes:
1. LIANA-based ligand-receptor interaction inference
2. CellPhoneDB statistical analysis
3. Spatial context analysis (if available)
4. Tensor decomposition for complex patterns
5. Comprehensive report of findings

Check the generated files for detailed interaction tables, visualizations, and the final analysis report.
</conclude>
