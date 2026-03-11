# Spatial Annotation

Annotate cell types and tissue niches in spatial transcriptomics data.

## Platform Applicability

**This workflow is for single-cell resolution platforms** (MERFISH, Xenium, CosMx, SeqFISH) where each observation corresponds to one cell.

**NOT for spot-based platforms** (Visium, Slide-seq, ST) where each spot contains multiple cells. For spot-based data, use the `spatial_deconvolution` skill instead.

**How to detect platform type**:
- **Single-cell resolution**: ~100–500 genes per panel, sub-cellular coordinates, technology names include MERFISH, Xenium, CosMx, SeqFISH
- **Spot-based**: ~18,000–33,000 genes (whole transcriptome), ~55µm spot diameter (Visium) or bead-based capture, technology names include Visium, Slide-seq, ST, 10x Spatial Gene Expression

---

## Workflow Overview

1. **Explore dataset structure** (always do this first)
2. Preprocess spatial data
3. Find and download scRNA-seq reference from CZI
4. Transfer cell type labels via Harmony integration
5. Annotate cell types using hierarchical approach
6. Run spatial clustering (UTAG) for tissue niches
7. Annotate tissue niches

---

## Step 1: Explore Dataset Structure

**ALWAYS start here.** Use `execute_python` to understand the data:

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/data.h5ad")

# Basic info
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var columns: {list(adata.var.columns)}")

# Check for spatial coordinates
if 'spatial' in adata.obsm:
    print(f"Spatial coords: {adata.obsm['spatial'].shape}")

# Check data range (normalized?)
print(f"X max: {adata.X.max():.2f}, min: {adata.X.min():.2f}")

# Sample/batch structure
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

## Step 2: Preprocess Spatial Data

```python
# Preprocess the spatial data
result = preprocess_spatial_data({
    "adata_path": "path/to/your/data.h5ad",
    "save_path": SAVE_PATH,
    "min_genes": 200,
    "min_cells": 3,
    "n_top_genes": 2000
})
print(result)
```

This performs:
- Quality control filtering
- Normalization and log transformation
- Highly variable gene selection
- PCA and UMAP computation
- QC metric visualization

---

## Step 3: Find Reference Dataset

Search CZI for matching scRNA-seq reference:

```python
# Search for relevant reference datasets
result = search_czi_datasets({
    "query": "human liver single cell RNA-seq",
    "organism": "Homo sapiens",
    "disease": "normal",
    "n_datasets": 5
})
print(result)
```

**Look for**:
- Same tissue/organ
- Same organism
- Healthy/normal condition (unless studying disease)
- Single-cell resolution (not spatial)

---

## Step 4: Download Reference Data

```python
# Download the selected reference dataset
result = download_czi_reference({
    "dataset_id": "chosen_dataset_id_from_step_3",
    "save_path": DATA_PATH
})
print(result)
```

---

## Step 5: Harmony Label Transfer

```python
# Transfer labels using Harmony integration
result = harmony_transfer_labels({
    "adata_path": os.path.join(SAVE_PATH, "processed_data.h5ad"),
    "ref_path": os.path.join(DATA_PATH, "reference_dataset.h5ad"),
    "save_path": SAVE_PATH,
    "batch_key": "sample",  # Adjust based on your data
    "label_key": "cell_type"  # Reference cell type column
})
print(result)
```

This performs:
- Data integration with Harmony
- Label transfer from reference to query
- Visualization of transferred annotations

---

## Step 6: Spatial Clustering (UTAG)

```python
# Run UTAG spatial clustering for tissue niches
result = run_utag_clustering({
    "adata_path": os.path.join(SAVE_PATH, "harmony_annotated.h5ad"),
    "save_path": SAVE_PATH,
    "resolution": 0.8
})
print(result)
```

This identifies spatial domains/tissue niches based on:
- Spatial coordinates
- Gene expression patterns
- Neighborhood relationships

---

## Step 7: Tissue Niche Annotation

```python
# Annotate tissue niches based on cell type composition
result = annotate_tissue_niches({
    "adata_path": os.path.join(SAVE_PATH, "utag_clustered.h5ad"),
    "cell_type_key": "predicted_cell_type",  # From Harmony transfer
    "save_path": SAVE_PATH
})
print(result)
```

This identifies functional tissue regions like:
- Tumor core vs. margin
- Immune niches
- Vascular regions
- Fibrotic areas

---

## Step 8: Final Summary

```python
# Generate final annotation summary
result = summarize_celltypes({
    "adata_path": os.path.join(SAVE_PATH, "tissue_niches.h5ad"),
    "celltype_key": "predicted_cell_type",
    "save_path": SAVE_PATH
})
print(result)

result = summarize_tissue_regions({
    "adata_path": os.path.join(SAVE_PATH, "tissue_niches.h5ad"),
    "region_key": "tissue_niche",
    "save_path": SAVE_PATH
})
print(result)
```

---

## Expected Outputs

1. **Processed data** with QC metrics and embeddings
2. **Cell type annotations** transferred from reference
3. **Spatial clusters** representing tissue niches
4. **Tissue niche annotations** with biological interpretation
5. **Summary statistics** and visualizations

---

## Common Issues

**No spatial coordinates found**:
- Check `adata.obsm` for coordinate columns
- Look for columns like `spatial`, `X_spatial`, `coordinates`

**Poor label transfer**:
- Reference and query may be from different tissues
- Check species compatibility
- Consider using different integration parameters

**Too many/few clusters**:
- Adjust UTAG resolution parameter
- Check data quality and preprocessing
- Consider biological context

---

## Advanced Options

**Custom marker genes**:
```python
# Use your own marker genes for annotation
markers = '{"T cells": ["CD3D", "CD2"], "B cells": ["MS4A1", "CD79A"]}'
result = annotate_cell_types({
    "adata_path": "data.h5ad",
    "marker_genes": markers,
    "save_path": SAVE_PATH
})
```

**Alternative integration methods**:
- Seurat integration (if available)
- Scanpy ingest
- Liger integration

---

## Quality Control

Always check:
- **QC plots** from preprocessing step
- **Label transfer accuracy** (known markers)
- **Spatial coherence** of clusters
- **Biological plausibility** of annotations

<conclude>
Spatial annotation workflow completed. Your data now has:
1. Cell type annotations transferred from reference
2. Spatial tissue niche identification
3. Comprehensive QC and visualizations
4. Summary statistics for both cell types and tissue regions

Check the generated files in your save path for detailed results and visualizations.
</conclude>
