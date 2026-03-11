"""
Modern function-based tools for SpatialAgent.

All tools use simple functions with @tool decorator instead of classes.
Organized by functional type: databases, analytics, interpretation, literature, coding, and foundry.

Note: Most tools are direct @tool decorated functions. Coding tools use creator functions
because they need initialization parameters (save_path, data_path).
"""

# Database & Reference Query Tools (direct @tool functions)
# Computational & Statistical Analysis Tools (direct @tool functions)
from .analytics import (
    aggregate_gene_voting,
    cell2location_mapping,
    cellphonedb_analysis,
    cellphonedb_degs_analysis,
    cellphonedb_filter,
    cellphonedb_plot,
    # CellPhoneDB tools
    cellphonedb_prepare,
    cellrank_fate_probabilities,
    cellrank_terminal_states,
    # scvi-tools spatial deconvolution
    destvi_deconvolution,
    gimvi_imputation,
    graphst_clustering,
    harmony_transfer_labels,
    infer_dynamics,
    liana_inference,
    liana_misty,
    liana_plot,
    liana_spatial,
    # LIANA tools
    liana_tensor,
    mofa_integration,
    multivi_integration,
    paga_trajectory,
    preprocess_spatial_data,
    run_utag_clustering,
    scanpy_bbknn,
    scanpy_ingest,
    # Scanpy tools
    scanpy_score_genes,
    # Trajectory inference
    scvelo_velocity,
    scvelo_velocity_embedding,
    # Spatial domain detection (SpaGCN, GraphST)
    spagcn_clustering,
    squidpy_centrality,
    squidpy_co_occurrence,
    squidpy_interaction_matrix,
    squidpy_ligrec,
    squidpy_nhood_enrichment,
    squidpy_ripley,
    squidpy_spatial_autocorr,
    # Squidpy tools
    squidpy_spatial_neighbors,
    stereoscope_deconvolution,
    summarize_celltypes,
    summarize_conditions,
    summarize_tissue_regions,
    tangram_evaluate,
    tangram_map_cells,
    # Tangram tools
    tangram_preprocess,
    tangram_project_annotations,
    tangram_project_genes,
    # Multimodal integration
    totalvi_integration,
)

# CodeAct: Python REPL and Bash (creator functions - need initialization)
from .coding import create_bash_tool, create_python_repl_tool
from .databases import (
    download_czi_reference,
    extract_czi_markers,
    query_celltype_genesets,
    query_disease_genes,
    query_tissue_expression,
    search_cellmarker2,
    search_czi_datasets,
    search_panglao,
    validate_genes_expression,
)

# Code inspection: Retrieve and adapt tool source code (direct @tool function)
from .foundry import inspect_tool_code

# LLM-Powered Interpretation Tools (direct @tool functions)
from .interpretation import (
    annotate_cell_types,
    annotate_tissue_niches,
    interpret_figure,
)

# Literature Research Tools (direct @tool functions)
from .literature import (
    extract_pdf_content,
    # query_scholar,  # Disabled - hangs due to Google Scholar rate limits
    # search_duckduckgo,  # Disabled - blocked on many networks, overlaps with academic search
    extract_url_content,
    fetch_supplementary_from_doi,
    query_arxiv,
    query_pubmed,
    search_semantic_scholar,
    web_search,  # Unified web search using Anthropic/OpenAI/Google server-side tools
)

# Subagent Tools (autonomous multi-step analysis)
from .subagent import (
    report_subagent,
    verification_subagent,
)

__all__ = [
    # Database Tools
    "search_panglao",
    "search_cellmarker2",
    "search_czi_datasets",
    "extract_czi_markers",
    "download_czi_reference",
    "query_tissue_expression",
    "query_celltype_genesets",
    "validate_genes_expression",
    "query_disease_genes",
    # Literature Research Tools
    "query_pubmed",
    "query_arxiv",
    "search_semantic_scholar",
    "web_search",
    # "query_scholar",  # Disabled
    # "search_duckduckgo",  # Disabled
    "extract_url_content",
    "extract_pdf_content",
    "fetch_supplementary_from_doi",
    # Analytics Tools
    "preprocess_spatial_data",
    "harmony_transfer_labels",
    "run_utag_clustering",
    "aggregate_gene_voting",
    "infer_dynamics",
    "summarize_conditions",
    "summarize_celltypes",
    "summarize_tissue_regions",
    # Tangram Tools
    "tangram_preprocess",
    "tangram_map_cells",
    "tangram_project_annotations",
    "tangram_project_genes",
    "tangram_evaluate",
    # CellPhoneDB Tools
    "cellphonedb_prepare",
    "cellphonedb_analysis",
    "cellphonedb_degs_analysis",
    "cellphonedb_filter",
    "cellphonedb_plot",
    # LIANA Tools
    "liana_tensor",
    "liana_inference",
    "liana_spatial",
    "liana_misty",
    "liana_plot",
    # Squidpy Tools
    "squidpy_spatial_neighbors",
    "squidpy_nhood_enrichment",
    "squidpy_co_occurrence",
    "squidpy_spatial_autocorr",
    "squidpy_ripley",
    "squidpy_centrality",
    "squidpy_interaction_matrix",
    "squidpy_ligrec",
    # scvi-tools Spatial Deconvolution
    "destvi_deconvolution",
    "cell2location_mapping",
    "stereoscope_deconvolution",
    "gimvi_imputation",
    # Spatial Domain Detection
    "spagcn_clustering",
    "graphst_clustering",
    # Scanpy Tools
    "scanpy_score_genes",
    "scanpy_ingest",
    "scanpy_bbknn",
    # Trajectory Inference
    "scvelo_velocity",
    "scvelo_velocity_embedding",
    "cellrank_terminal_states",
    "cellrank_fate_probabilities",
    "paga_trajectory",
    # Multimodal Integration
    "totalvi_integration",
    "multivi_integration",
    "mofa_integration",
    # Interpretation Tools
    "annotate_cell_types",
    "annotate_tissue_niches",
    "interpret_figure",
    # Subagent Tools
    "report_subagent",
    "verification_subagent",
    # CodeAct (creator functions)
    "create_python_repl_tool",
    "create_bash_tool",
    # Support
    "inspect_tool_code",
]
