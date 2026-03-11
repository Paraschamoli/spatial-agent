# Gene Panel Design

Design targeted gene panels for spatial transcriptomics platforms.

## Platform Applicability

**This workflow is for targeted platforms** with limited gene panels:
- **MERFISH**: ~500-1000 genes
- **Xenium**: ~300-500 genes
- **CosMx**: ~1000 genes
- **SeqFISH**: ~200-500 genes
- **NanoString GeoMx**: ~50-800 genes

**NOT for whole-transcriptome platforms** (Visium, Slide-seq, ST).

---

## Workflow Overview

1. **Define experimental goals** (tissue, cell types, biological questions)
2. **Literature search** for relevant markers and pathways
3. **Database mining** for cell type and disease markers
4. **Marker aggregation** using voting system
5. **Expression validation** in reference datasets
6. **Panel optimization** and final selection

---

## Step 1: Define Experimental Goals

**ALWAYS start here.** Define your requirements:

```python
# Example: Mouse prostate cancer panel
tissue = "prostate"
organism = "mouse"
disease = "cancer"
target_cell_types = ["epithelial", "immune", "stromal", "endothelial"]
panel_size = 50  # Target number of genes
biological_questions = [
    "tumor state",
    "immune process",
    "tissue context"
]
```

**Consider**:
- Tissue/organ of interest
- Disease vs. normal
- Key cell types to capture
- Biological processes/pathways
- Platform constraints (gene number)

---

## Step 2: Literature Search

Search PubMed for relevant marker genes:

```python
# Search for tissue-specific markers
result = query_pubmed({
    "query": f"{tissue} {organism} marker genes single cell RNA-seq",
    "max_papers": 5
})
print(result)

# Search for disease-specific markers
result = query_pubmed({
    "query": f"{disease} {tissue} biomarkers single cell",
    "max_papers": 5
})
print(result)

# Search for pathway markers
for process in biological_questions:
    result = query_pubmed({
        "query": f"{process} {tissue} markers spatial transcriptomics",
        "max_papers": 3
    })
    print(result)
```

---

## Step 3: Database Mining

**Cell type markers from PanglaoDB**:

```python
# Get markers for each target cell type
all_markers = {}
for cell_type in target_cell_types:
    result = search_panglao({
        "cell_types": cell_type,
        "organism": "Mm" if organism == "mouse" else "Hs",
        "tissue": tissue,
        "min_specificity": 0.7,
        "min_sensitivity": 0.5
    })
    print(f"\n{cell_type} markers:")
    print(result)

    # Extract genes from result (simplified - would need parsing)
    # For now, note the result structure
    all_markers[cell_type] = result
```

**Disease-associated genes**:

```python
# Get disease-related genes
result = query_disease_genes({
    "disease": f"{disease} {tissue}",
    "organism": organism
})
print("Disease-associated genes:")
print(result)
```

**Cell type gene sets**:

```python
# Get comprehensive gene sets for cell types
for cell_type in target_cell_types:
    result = query_celltype_genesets({
        "cell_type": cell_type,
        "organism": organism
    })
    print(f"\n{cell_type} gene sets:")
    print(result)
```

---

## Step 4: Marker Aggregation

Collect all candidate genes and use voting:

```python
# Collect candidate genes from all sources
import json

# Example structure - would need to parse actual results
candidate_genes = {
    "epithelial_markers": ["KRT8", "KRT18", "EPCAM", "KRT5", "KRT14"],
    "immune_markers": ["CD3D", "CD2", "MS4A1", "CD79A", "LYZ"],
    "stromal_markers": ["COL1A1", "COL3A1", "ACTA2", "PDGFRA"],
    "endothelial_markers": ["PECAM1", "VWF", "CDH5", "ESAM"],
    "disease_markers": ["MKI67", "TOP2A", "CCNB1"],
    "literature_markers": ["AR", "KLK3", "MSMB"]
}

# Use voting aggregation
result = aggregate_gene_voting({
    "gene_lists": json.dumps(candidate_genes),
    "method": "vote",
    "min_support": 0.3  # Genes must appear in 30% of lists
})
print(result)
```

---

## Step 5: Expression Validation

**Validate in reference datasets**:

```python
# Find relevant reference dataset
result = search_czi_datasets({
    "query": f"{tissue} {organism} single cell",
    "organism": "Homo sapiens" if organism == "human" else "Mus musculus",
    "disease": disease,
    "n_datasets": 3
})
print(result)

# Download and validate expression
# (This would use the selected dataset ID)
result = validate_genes_expression({
    "genes": "KRT8,KRT18,EPCAM,CD3D,MS4A1,COL1A1,PECAM1,MKI67,AR,KLK3",
    "data_path": os.path.join(DATA_PATH, "reference_dataset.h5ad")
})
print(result)
```

**Tissue expression validation**:

```python
# Check tissue-specific expression
for gene in ["KRT8", "CD3D", "COL1A1", "PECAM1", "MKI67"]:
    result = query_tissue_expression({
        "genes": gene,
        "tissue": tissue,
        "organism": organism
    })
    print(f"{gene} in {tissue}:")
    print(result)
```

---

## Step 6: Panel Optimization

**Final gene selection**:

```python
# Example final panel based on validation
final_panel = [
    # Epithelial/Tumor
    "EPCAM", "KRT8", "KRT18", "KRT5", "MKI67", "TOP2A",
    # Prostate specific
    "AR", "KLK3", "MSMB", "KLK2", "FKBP5",
    # Immune
    "CD3D", "CD2", "MS4A1", "CD79A", "LYZ", "CD68",
    # Stromal/Fibroblast
    "COL1A1", "COL3A1", "ACTA2", "PDGFRA", "FAP",
    # Endothelial
    "PECAM1", "VWF", "CDH5", "ESAM", "ENG",
    # Signaling/Pathways
    "TGFB1", "IL6", "CXCL12", "VEGFA", "PDGFB"
]

print(f"Final panel ({len(final_panel)} genes):")
for i, gene in enumerate(final_panel, 1):
    print(f"{i:2d}. {gene}")

# Save panel
import os
os.makedirs(SAVE_PATH, exist_ok=True)
with open(os.path.join(SAVE_PATH, "final_gene_panel.txt"), 'w') as f:
    f.write("# Final Gene Panel for Spatial Transcriptomics\n")
    f.write(f"# Tissue: {tissue}\n")
    f.write(f"# Organism: {organism}\n")
    f.write(f"# Disease: {disease}\n")
    f.write(f"# Target size: {panel_size}\n")
    f.write(f"# Actual size: {len(final_panel)}\n\n")
    for gene in final_panel:
        f.write(f"{gene}\n")
```

---

## Step 7: Generate Report

```python
# Generate comprehensive panel design report
result = report_subagent({
    "task_description": f"Design {panel_size}-gene panel for {organism} {tissue} {disease} analysis",
    "data_context": f"Target cell types: {', '.join(target_cell_types)}\nBiological processes: {', '.join(biological_questions)}",
    "save_path": SAVE_PATH,
    "report_type": "detailed"
})
print(result)
```

---

## Expected Outputs

1. **Literature search results** with relevant papers and markers
2. **Database query results** from PanglaoDB, CellMarker, etc.
3. **Aggregated gene list** with voting scores
4. **Expression validation** in reference datasets
5. **Final gene panel** with biological justification
6. **Comprehensive design report**

---

## Common Issues

**Too many candidate genes**:
- Increase specificity thresholds
- Focus on most relevant cell types
- Prioritize disease-specific markers

**Poor expression validation**:
- Check species compatibility
- Consider alternative markers
- Validate in multiple datasets

**Panel size constraints**:
- Prioritize by biological relevance
- Use hierarchical selection
- Consider multiplexed panels

---

## Advanced Options

**Pathway-focused panels**:
- Include signaling pathway genes
- Add transcription factors
- Consider secreted factors

**Quality control genes**:
- Housekeeping genes for normalization
- Negative control genes
- Platform-specific controls

**Multiplexing strategy**:
- Split large panels into sub-panels
- Overlapping genes for validation
- Sequential staining approaches

---

## Quality Control

Always validate:
- **Gene expression** in target tissue
- **Cell type specificity** of markers
- **Platform compatibility** (probe design)
- **Biological relevance** to research questions

<conclude>
Gene panel design completed. Your final panel includes:
1. Literature-validated marker genes
2. Database-supported cell type markers
3. Disease-associated genes
4. Tissue-specific expression validation
5. Optimized gene selection for your platform

The panel file and comprehensive design report have been saved to your specified directory. Review the biological justification and validation results before proceeding to experimental implementation.
</conclude>
