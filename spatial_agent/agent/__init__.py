# SpatialAgent - LangGraph-based agent
from .make_llm import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    LOCAL_EMBEDDING_MODELS,
    CostCallback,
    LocalEmbeddings,
    get_effective_embedding_model,
    make_llm,
    make_llm_emb,
    make_llm_emb_local,
)

# from .make_tool import make_tool  # Commented out to avoid circular import - import directly from .make_tool
from .make_prompt import AgentPrompts
from .spatialagent import SpatialAgent

# Shared configuration for subagents and tool selectors
# This is set by the main agent and used by subcomponents
_agent_config = {
    "model": None,  # Will be set by SpatialAgent
    "llm": None,  # Will be set by SpatialAgent
}


def set_agent_model(model_name: str, llm=None):
    """Set the model name for subagents to use."""
    _agent_config["model"] = model_name
    _agent_config["llm"] = llm


def get_agent_model() -> str:
    """Get the model name set by the main agent."""
    return _agent_config["model"] or DEFAULT_CLAUDE_MODEL


def get_agent_llm():
    """Get the LLM instance set by the main agent."""
    return _agent_config["llm"]
