# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""Spatial Agent - A Bindu AI Agent for Spatial Transcriptomics Analysis."""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

from .agents import run_spatial_analysis

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Any = None
_initialized = False
_init_lock = asyncio.Lock()

# Setup logging
_logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """Load agent config from `agent_config.json` or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    return {
        "name": "spatial-agent",
        "description": "AI-powered spatial transcriptomics analysis assistant",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


class SpatialDiagnosticsAgent:
    """Spatial Agent wrapper following the research-agent pattern."""

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize spatial agent with model name."""
        self.model_name = model_name

        # Get API key from environment (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            print(f"✅ Using OpenRouter model: {model_name}")
        else:
            error_msg = (
                "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
                "Get your key from: https://openrouter.ai/keys"
            )
            raise ValueError(error_msg)

    async def arun(self, messages: list[dict[str, str]]) -> str:
        """Run the agent with the given messages - matches research-agent pattern."""
        # Extract data description from messages
        data_description = ""
        for message in messages:
            if message.get("role") == "user":
                data_description = message.get("content", "")
                break

        if not data_description:
            return "Error: No data description provided in the user message."

        try:
            # Run the spatial analysis pipeline
            final_analysis = await run_spatial_analysis(data_description, self.model_name)

            # Format the response
            response = f"""### Spatial Analysis Plan:

{final_analysis}

---
*Analysis performed by AI Spatial Transcriptomics Agent*
*Specialist inputs: Annotation Specialist, Communication Analyst, Spatial Domain Expert*
*Model: {self.model_name}*
"""

            return response

        except Exception as e:
            error_msg = f"Error during spatial analysis: {e!s}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return error_msg


async def initialize_agent() -> None:
    """Initialize spatial agent with proper model."""
    global agent

    # Get API key and model from environment
    model_name = os.getenv("MODEL_NAME", "gpt-4o")

    # Get API key from environment (only supports OpenRouter)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_api_key:
        agent = SpatialDiagnosticsAgent(model_name)
        print(f"✅ Using model: {model_name}")
    else:
        error_msg = (
            "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
            "Get your key from: https://openrouter.ai/keys"
        )
        raise ValueError(error_msg)

    print("✅ Spatial Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages - matches research-agent pattern."""
    global agent
    if not agent:
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response - matches research-agent pattern
    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization - matches research-agent pattern."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Spatial Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Spatial Agent resources...")


def main():
    """Run the main entry point for the Spatial Agent."""
    parser = argparse.ArgumentParser(description="Bindu Spatial Transcriptomics Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929"),
        help="Model name to use (env: MODEL_NAME, default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Spatial Transcriptomics Agent - AI Spatial Biology Analysis")
    print("🔬 Capabilities: Cell type annotation, communication analysis, spatial domain detection")
    print("👥 Specialist Team: Annotation Specialist, Communication Analyst, Spatial Domain Expert")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Spatial Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3774')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Spatial Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


# Bindufy and start the agent server
if __name__ == "__main__":
    main()
