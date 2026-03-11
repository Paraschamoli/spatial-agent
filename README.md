<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">spatial-agent</h1>

<p align="center">
  <strong>A Bindu AI agent for spatial transcriptomics analysis</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/spatial-agent/actions/workflows/main.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/spatial-agent/main.yml?branch=main" alt="Build status">
  </a>
  <a href="https://img.shields.io/github/license/Paraschamoli/spatial-agent">
    <img src="https://img.shields.io/github/license/Paraschamoli/spatial-agent" alt="License">
  </a>
</p>

---

## 📖 Overview

A specialized Bindu AI agent for comprehensive spatial transcriptomics analysis. Built on the [Bindu Agent Framework](https://github.com/getbindu/bindu) for the Internet of Agents.

**Key Capabilities:**
- � **Cell Type Annotation**: Automated cell type identification using reference datasets (PanglaoDB, CellMarker, CZI)
- 🤝 **Cell-Cell Communication**: Ligand-receptor interaction analysis with CellPhoneDB and LIANA
- 🗺️ **Spatial Domain Detection**: Spatial clustering and tissue architecture analysis (UTAG, SpaGCN, GraphST)
- 🧬 **Gene Panel Design**: Targeted gene panel design for spatial platforms
- 📊 **Spatial Mapping**: Integration of scRNA-seq with spatial data using Tangram
- 🔄 **Trajectory Inference**: RNA velocity and fate mapping analysis (scVelo, CellRank)
- 🔗 **Multimodal Integration**: Multi-omics data integration (TotalVI, MultiVI, MOFA+)
- � **Literature Research**: PubMed, arXiv, and Semantic Scholar search capabilities
- 🗄️ **Database Query**: Access to biological databases and reference datasets

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/spatial-agent.git
cd spatial-agent

# Create virtual environment
uv venv --python 3.12.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
```

### Configuration

Edit `.env` and add your API key:

| Key | Get It From | Required |
|-----|-------------|----------|
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/keys) | ✅ Yes |

### Run the Agent

```bash
# Start the agent
uv run python -m spatial_agent

# Agent will be available at http://localhost:3773
```

### Github Setup

```bash
# Initialize git repository and commit your code
git init -b main
git add .
git commit -m "Initial commit"

# Create repository on GitHub and push (replace with your GitHub username)
gh repo create Paraschamoli/spatial-agent --public --source=. --remote=origin --push
```

---

## 💡 Usage

### Example Queries

```bash
# Example query 1: Cell type annotation
"I have a spatial transcriptomics dataset from human brain tissue analyzed with 10x Visium. I need to identify neuronal and glial cell types using PanglaoDB markers and create spatial visualizations."

# Example query 2: Cell-cell communication analysis
"Analyze ligand-receptor interactions between tumor cells and immune cells in my breast cancer spatial dataset using CellPhoneDB. Focus on immune checkpoint pathways."

# Example query 3: Spatial domain detection
"Perform spatial clustering on my lung tissue dataset to identify distinct anatomical regions using UTAG and SpaGCN algorithms."
```

### Input Formats

**Plain Text:**
```
I have a spatial transcriptomics dataset from [tissue type] with [number] spots. I need to [specific analysis task] using [tools/methods].
```

**JSON:**
```json
{
  "tissue_type": "human brain",
  "platform": "10x Visium",
  "spots": 5000,
  "analysis_goal": "cell type annotation",
  "focus_genes": ["GFAP", "SLC17A7", "GAD1"],
  "tools_requested": ["PanglaoDB", "CellPhoneDB", "UTAG"]
}
```

### Output Structure

The agent returns structured output with:
- **Analysis Plan**: Step-by-step workflow for spatial analysis
- **Tool Recommendations**: Specific tools and parameters for each analysis step
- **Cell Type Annotations**: Identified cell types with marker genes
- **Communication Networks**: Ligand-receptor interaction networks
- **Spatial Domains**: Detected tissue regions and clusters
- **Visualization Guidance**: Recommended plots and spatial visualizations
- **Code Examples**: Ready-to-use code snippets for analysis implementation

---

## 🔌 API Usage

The agent exposes a RESTful API when running. Default endpoint: `http://localhost:3773` 

### Quick Start

For complete API documentation, request/response formats, and examples, visit:

📚 **[Bindu API Reference - Send Message to Agent](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)**


### Additional Resources

- 📖 [Full API Documentation](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)
- 📦 [Postman Collections](https://github.com/GetBindu/Bindu/tree/main/postman/collections)
- 🔧 [API Reference](https://docs.getbindu.com)

---

## 🎯 Skills

### spatial (v1.0.0)

**Primary Capability:**
- Comprehensive spatial transcriptomics analysis workflow planning and execution
- Integration of multiple specialized analysis tools (CellPhoneDB, LIANA, UTAG, Tangram, etc.)
- Expert consultation from Annotation Specialist, Communication Analyst, and Spatial Domain Expert

**Features:**
- **Multi-Specialist Integration**: Combines expertise from three specialized AI agents
- **72 Specialized Tools**: Access to comprehensive spatial biology analysis toolkit
- **Database Integration**: Direct access to PanglaoDB, CellMarker, CZI, and PubMed
- **Literature Research**: Automated literature search and citation management
- **Code Generation**: Ready-to-use Python code for spatial analysis
- **Visualization Planning**: Expert guidance on spatial data visualization

**Best Used For:**
- Complete spatial transcriptomics analysis workflows
- Cell type annotation and marker discovery
- Cell-cell communication network analysis
- Spatial domain detection and tissue architecture analysis
- Multi-omics integration projects
- Research planning and experimental design

**Not Suitable For:**
- Real-time clinical diagnostics (research-focused only)
- Non-biological data analysis
- Simple data visualization without analysis context

**Performance:**
- Average processing time: ~30-60 seconds for comprehensive analysis plan
- Max concurrent requests: 10
- Memory per request: ~500MB

---

## 🐳 Docker Deployment

### Local Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Agent will be available at http://localhost:3773
```

### Docker Configuration

The agent runs on port `3773` and requires:
- `OPENROUTER_API_KEY` environment variable

Configure this in your `.env` file before running.

### Production Deployment

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🌐 Deploy to bindus.directory

Make your agent discoverable worldwide and enable agent-to-agent collaboration.

### Setup GitHub Secrets

```bash
# Authenticate with GitHub
gh auth login

# Set deployment secrets
gh secret set BINDU_API_TOKEN --body "<your-bindu-api-key>"
gh secret set DOCKERHUB_TOKEN --body "<your-dockerhub-token>"
```

Get your keys:
- **Bindu API Key**: [bindus.directory](https://bindus.directory) dashboard
- **Docker Hub Token**: [Docker Hub Security Settings](https://hub.docker.com/settings/security)

### Deploy

```bash
# Push to trigger automatic deployment
git push origin main
```

GitHub Actions will automatically:
1. Build your agent
2. Create Docker container
3. Push to Docker Hub
4. Register on bindus.directory

---

## 🛠️ Development

### Project Structure

```
spatial-agent/
├── spatial_agent/
│   ├── tools/                      # 72 spatial analysis tools
│   │   ├── databases.py           # Database query tools
│   │   ├── analytics.py           # Computational analysis tools
│   │   ├── interpretation.py      # LLM-powered interpretation tools
│   │   ├── literature.py          # Literature research tools
│   │   ├── coding.py              # Python/Bash execution tools
│   │   ├── subagent.py            # Autonomous analysis agents
│   │   ├── foundry.py             # Code inspection tools
│   │   ├── utils.py               # Utility functions
│   │   └── __init__.py
│   ├── agent/                     # Core agent components
│   │   ├── spatialagent.py        # Main SpatialAgent class
│   │   ├── make_llm.py            # LLM factory (OpenRouter compatible)
│   │   ├── make_prompt.py         # System prompts
│   │   ├── tool_system.py         # Tool management system
│   │   ├── skills.py              # Skill template manager
│   │   └── utils.py               # Agent utilities
│   ├── skills/                    # Analysis skill templates
│   │   ├── annotation.md
│   │   ├── cell_cell_communication.md
│   │   ├── cell_deconvolution.md
│   │   └── [other skill templates]
│   ├── main.py                    # Agent entry point
│   ├── agents.py                  # Multi-specialist agents
│   ├── hooks.py                   # Agent lifecycle hooks
│   ├── agent_config.json          # Agent configuration
│   └── __init__.py
├── test_tools_access.py           # Tool verification script
├── .env.example                   # Environment variables template
├── pyproject.toml                 # Dependencies and project config
└── README.md
```

### Running Tests

```bash
# Verify all tools and imports are working
uv run python test_tools_access.py

# Expected output: 🎉 All tests passed! Agent can access tools and skills.
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Run linters
uv run ruff check .

# Format + lint + test
uv run ruff format . && uv run ruff check . && uv run python test_tools_access.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run -a
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature` 
3. Commit your changes: `git commit -m 'Add amazing feature'` 
4. Push to the branch: `git push origin feature/amazing-feature` 
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Powered by Bindu

Built with the [Bindu Agent Framework](https://github.com/getbindu/bindu)

**Why Bindu?**
- 🌐 **Internet of Agents**: A2A, AP2, X402 protocols for agent collaboration
- ⚡ **Zero-config setup**: From idea to production in minutes
- 🛠️ **Production-ready**: Built-in deployment, monitoring, and scaling

**Build Your Own Agent:**
```bash
uvx cookiecutter https://github.com/getbindu/create-bindu-agent.git
```

---

## 📚 Resources

- 📖 [Full Documentation](https://Paraschamoli.github.io/spatial-agent/)
- 💻 [GitHub Repository](https://github.com/Paraschamoli/spatial-agent/)
- 🐛 [Report Issues](https://github.com/Paraschamoli/spatial-agent/issues)
- 💬 [Join Discord](https://discord.gg/3w5zuYUuwt)
- 🌐 [Agent Directory](https://bindus.directory)
- 📚 [Bindu Documentation](https://docs.getbindu.com)

---

<p align="center">
  <strong>Built with 💛 by the team from Amsterdam 🌷</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/spatial-agent">⭐ Star this repo</a> •
  <a href="https://discord.gg/3w5zuYUuwt">💬 Join Discord</a> •
  <a href="https://bindus.directory">🌐 Agent Directory</a>
</p>

#   s p a t i a l - a g e n t  
 