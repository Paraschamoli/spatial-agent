"""Utility functions for agent system."""

import importlib
import inspect


def load_all_tools(save_path: str = "./experiments", data_path: str = "./data"):
    """
    Auto-discover and load all tool functions from tool modules.

    Args:
        save_path: Path for experiment outputs (for coding tools)
        data_path: Path to reference data (for coding tools)

    Returns:
        list: List of LangChain tool instances
    """
    # Configure tool paths
    from spatial_agent.tools.coding import configure_coding_tools
    from spatial_agent.tools.databases import configure_database_tools
    from spatial_agent.tools.analytics import configure_analytics_tools
    from spatial_agent.tools.interpretation import configure_interpretation_tools
    from spatial_agent.tools.subagent import configure_subagent_tools
    configure_coding_tools(save_path, data_path)
    configure_database_tools(data_path)
    configure_analytics_tools(save_path)
    configure_interpretation_tools(save_path)
    configure_subagent_tools(save_path)

    # Map module names to their tool modules
    tool_modules = {
        "database": "spatial_agent.tools.databases",
        "literature": "spatial_agent.tools.literature",
        "analytics": "spatial_agent.tools.analytics",
        "interpretation": "spatial_agent.tools.interpretation",
        "support_tools": "spatial_agent.tools.foundry",
        "coding": "spatial_agent.tools.coding",
        "subagent": "spatial_agent.tools.subagent",
    }

    all_tools = []

    for category, module_path in tool_modules.items():
        try:
            module = importlib.import_module(module_path)

            # Find all functions decorated with @tool
            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "name") and hasattr(obj, "description"):
                    # This is a LangChain tool
                    all_tools.append(obj)
                    print(f"  Loaded: {obj.name} ({category})")

        except Exception as e:
            print(f"Warning: Could not load tools from {module_path}: {e}")

    return all_tools
