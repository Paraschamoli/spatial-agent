# Tangram Mapping Validation

Validate and troubleshoot Tangram spatial mappings using cross-validation, diagnostic plots, and quality metrics.

## Why Validate?

- Ensure mapping is biologically meaningful
- Identify problematic genes or cell types
- Compare different mapping strategies
- Build confidence before downstream analysis

---

## Validation Approaches

1. **Training score analysis**: How well do training genes match?
2. **Test gene prediction**: Can we predict held-out genes?
3. **Cross-validation**: Systematic holdout testing
4. **Biological validation**: Do patterns match known biology?

---

## Step 1: Basic Quality Check

After running `tangram_map_cells`, check training scores:

```python
import scanpy as sc
import tangram as tg

ad_map = sc.read_h5ad("experiments/tangram_mapping.h5ad")

# Training gene scores
train_df = ad_map.uns["train_genes_df"]
print(f"Average training score: {train_df['train_score'].mean():.3f}")
print(f"Median training score: {train_df['train_score'].median():.3f}")

# Score distribution
print("\nScore distribution:")
print(train_df['train_score'].describe())

# Top and bottom genes
print("\nTop 10 genes:")
print(train_df.head(10))
print("\nBottom 10 genes:")
print(train_df.tail(10))
```

**Interpretation**:
- Average > 0.7: Good mapping
- Average 0.5-0.7: Moderate, may need tuning
- Average < 0.5: Poor, check data compatibility

---

## Step 2: Training Diagnostic Plots

**Tool**: `tangram_evaluate`

Or manually generate plots:

```python
import tangram as tg
import matplotlib.pyplot as plt

# 4-panel diagnostic plot
tg.plot_training_scores(ad_map, bins=20, alpha=0.5)
plt.savefig("training_diagnostics.png", dpi=150, bbox_inches="tight")
```

**Panels explained**:
1. **Score histogram**: Distribution of gene scores
2. **Score vs SC sparsity**: Low sparsity genes should score high
3. **Score vs SP sparsity**: Sparse spatial genes often score low
4. **Score vs sparsity diff**: Large diff = dropout mismatch

---

## Step 3: Test Gene Evaluation

Project genes and evaluate predictions:

```python
# Project genes
ad_ge = tg.project_genes(ad_map, adata_sc)
ad_ge.write_h5ad("experiments/tangram_projected.h5ad")

# Compare with spatial data
df_genes = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)

# Separate training and test
train_genes = df_genes[df_genes["is_training"] == True]
test_genes = df_genes[df_genes["is_training"] == False]

print(f"Training genes: {len(train_genes)}, avg score: {train_genes['score'].mean():.3f}")
print(f"Test genes: {len(test_genes)}, avg score: {test_genes['score'].mean():.3f}")
```

---

## Step 4: Cross-Validation

Systematic holdout testing for robust validation:

```python
import numpy as np
from sklearn.model_selection import KFold

# Load data
adata_sc = sc.read_h5ad("scrna.h5ad")
adata_sp = sc.read_h5ad("spatial.h5ad")

# Get shared genes
shared_genes = list(set(adata_sc.var_names) & set(adata_sp.var_names))
print(f"Shared genes: {len(shared_genes)}")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(shared_genes)):
    print(f"\nFold {fold + 1}/5")
    
    train_genes = [shared_genes[i] for i in train_idx]
    test_genes = [shared_genes[i] for i in test_idx]
    
    # Preprocess with training genes only
    tangram_preprocess(
        adata_sc_path="scrna.h5ad",
        adata_sp_path="spatial.h5ad",
        marker_genes=train_genes,
        cell_type_key="cell_type"
    )
    
    # Map cells
    ad_map = tg.map_cells_to_space(
        sc.read_h5ad("experiments/tangram_sc_prep.h5ad"),
        sc.read_h5ad("experiments/tangram_sp_prep.h5ad"),
        mode="cells",
        device="cuda:0"
    )
    
    # Project and evaluate test genes
    ad_ge = tg.project_genes(ad_map, adata_sc)
    df_test = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)
    test_scores = df_test[df_test['gene'].isin(test_genes)]['score']
    
    cv_scores.append(test_scores.mean())
    print(f"Test score: {test_scores.mean():.3f}")

print(f"\nCross-validation score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

---

## Step 5: Biological Validation

Check if mapped cell types match expected spatial patterns:

```python
import matplotlib.pyplot as plt

# Load mapping results
ad_map = sc.read_h5ad("experiments/tangram_mapping.h5ad")
adata_sp = sc.read_h5ad("spatial.h5ad")

# Get cell type predictions per spot
cell_type_predictions = ad_map.obsm['cell_type_predictions']

# Visualize major cell types
major_types = ['T cell', 'B cell', 'Macrophage', 'Fibroblast']
for ct in major_types:
    if ct in cell_type_predictions.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.spatial(adata_sp, color=ct, ax=ax, show=False, 
                     title=f"{ct} abundance (Tangram)")
        plt.savefig(f"tangram_{ct.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()
```

---

## Step 6: Compare Mapping Strategies

Test different Tangram configurations:

```python
configs = [
    {"mode": "cells", "name": "cells"},
    {"mode": "clusters", "name": "clusters"},
    {"mode": "constrained", "target_count": 1000, "name": "constrained"}
]

results = {}

for config in configs:
    print(f"\nTesting: {config['name']}")
    
    # Map with specific configuration
    if config["name"] == "constrained":
        ad_map = tg.map_cells_to_space(
            adata_sc, adata_sp, mode="constrained",
            target_count=config["target_count"], device="cuda:0"
        )
    else:
        ad_map = tg.map_cells_to_space(
            adata_sc, adata_sp, mode=config["mode"], device="cuda:0"
        )
    
    # Evaluate
    ad_ge = tg.project_genes(ad_map, adata_sc)
    df_eval = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)
    
    results[config["name"]] = {
        "train_score": df_eval[df_eval["is_training"]]["score"].mean(),
        "test_score": df_eval[~df_eval["is_training"]]["score"].mean()
    }
    
    print(f"  Train: {results[config['name']]['train_score']:.3f}")
    print(f"  Test: {results[config['name']]['test_score']:.3f}")

# Summary
print("\n=== Summary ===")
for name, scores in results.items():
    print(f"{name}: Train={scores['train_score']:.3f}, Test={scores['test_score']:.3f}")
```

---

## Common Issues and Solutions

### Low Training Scores (< 0.5)

**Possible causes**:
- Gene name mismatch between datasets
- Different species (mouse vs human)
- Poor data quality
- Incompatible cell type annotations

**Solutions**:
```python
# Check gene name overlap
sc_genes = set(adata_sc.var_names)
sp_genes = set(adata_sp.var_names)
overlap = sc_genes & sp_genes
print(f"Gene overlap: {len(overlap)}/{len(sc_genes)} ({len(overlap)/len(sc_genes):.1%})")

# Convert mouse to human genes if needed
import gget
mouse_genes = [g for g in sc_genes if g[0].isupper()]
human_equiv = gget.orthologs(mouse_genes, source_species="mouse", target_species="human")
```

### Large Score Variance

**Issue**: Some genes score very high, others very low

**Check sparsity patterns**:
```python
# Analyze gene sparsity
train_df = ad_map.uns["train_genes_df"]
low_sc_sparsity = train_df[train_df['sc_sparsity'] < 0.1]
high_sp_sparsity = train_df[train_df['sp_sparsity'] > 0.9]

print(f"Low SC sparsity genes: {len(low_sc_sparsity)}")
print(f"High SP sparsity genes: {len(high_sp_sparsity)}")
```

### Poor Test Gene Performance

**Issue**: Training genes score well but test genes don't

**Solutions**:
- Use more diverse training genes (not just markers)
- Increase training epochs
- Check for batch effects
- Consider alternative mapping modes

---

## Output Files

- `training_diagnostics.png` - 4-panel diagnostic plot
- `tangram_{celltype}.png` - Spatial predictions for each cell type
- `cv_results.csv` - Cross-validation scores
- `mapping_comparison.csv` - Different configuration results

---

## Tips

1. **Always validate** before biological interpretation
2. **Use cross-validation** for robust performance estimates
3. **Check biological plausibility** of spatial patterns
4. **Compare multiple configurations** to find optimal setup
5. **Document issues** for reproducibility
