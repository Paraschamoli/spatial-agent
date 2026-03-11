"""Spatial Analysis Agents using LangChain."""

import asyncio
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class SpatialAnalysisAgent:
    """Base class for spatial analysis agents."""

    def __init__(self, data_description: str, role: str, model_name: str = "gpt-4o"):
        """Initialize spatial analysis agent with data description, role, and model."""
        self.data_description = data_description
        self.role = role

        # Get API key from environment (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API key with OpenAI client
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        else:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable.")

        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create role-specific prompt template."""
        templates = {
            "Annotation Specialist": """
                You are a spatial transcriptomics annotation specialist. 
                You will receive a description of spatial transcriptomics data.
                Task: Analyze the data description and provide guidance on cell type annotation strategies.
                Focus: Identify appropriate reference datasets, annotation methods, and quality control measures.
                Recommendation: Suggest specific tools and workflows for accurate cell type identification.
                Please provide a detailed analysis plan for cell type annotation.

                Data Description: {data_description}
            """,
            "Communication Analyst": """
                You are a cell-cell communication analysis specialist.
                You will receive a description of spatial transcriptomics data.
                Task: Analyze the data description and provide guidance on communication analysis.
                Focus: Identify appropriate methods for ligand-receptor interaction analysis and spatial communication.
                Recommendation: Suggest specific tools for CellPhoneDB, LIANA, and spatial communication analysis.
                Please provide a detailed analysis plan for cell-cell communication.

                Data Description: {data_description}
            """,
            "Spatial Domain Expert": """
                You are a spatial domain detection specialist.
                You will receive a description of spatial transcriptomics data.
                Task: Analyze the data description and provide guidance on spatial domain identification.
                Focus: Identify appropriate methods for spatial clustering, domain detection, and tissue architecture analysis.
                Recommendation: Suggest specific tools for UTAG, SpaGCN, and spatial pattern analysis.
                Please provide a detailed analysis plan for spatial domain detection.

                Data Description: {data_description}
            """,
        }

        return PromptTemplate.from_template(templates[self.role])

    async def run(self) -> str:
        """Run the agent analysis."""
        print(f"{self.role} is running...")
        try:
            response = await self.chain.ainvoke({"data_description": self.data_description})
            return response
        except Exception as e:
            print(f"Error occurred in {self.role}: {e}")
            return f"Error: {e!s}"


class MultidisciplinarySpatialTeam:
    """Agent that synthesizes reports from multiple spatial analysis specialists."""

    def __init__(
        self, 
        annotation_report: str, 
        communication_report: str, 
        domain_report: str, 
        model_name: str = "gpt-4o"
    ):
        """Initialize multidisciplinary team with specialist reports."""
        self.annotation_report = annotation_report
        self.communication_report = communication_report
        self.domain_report = domain_report
        self.model_name = model_name

        # Create synthesis prompt
        synthesis_prompt = f"""
            You are a multidisciplinary team of spatial transcriptomics experts.
            You will receive analysis from an Annotation Specialist, Communication Analyst, and Spatial Domain Expert.
            Task: Review the three specialist reports and create a comprehensive spatial analysis plan.
            Focus: Integrate recommendations into a cohesive workflow that addresses all aspects of spatial analysis.
            Please provide a detailed, integrated analysis plan with specific tool recommendations and workflow steps.

            Annotation Specialist Report: {annotation_report}

            Communication Analyst Report: {communication_report}

            Spatial Domain Expert Report: {domain_report}
        """

        # Create the model using the same pattern as specialist agents
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        else:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable.")

        # Create synthesis chain
        self.chain = PromptTemplate.from_template(synthesis_prompt) | self.model | StrOutputParser()

    async def run(self) -> str:
        """Run the multidisciplinary analysis."""
        print("MultidisciplinarySpatialTeam is running...")
        try:
            response = await self.chain.ainvoke({})
            return response
        except Exception as e:
            print(f"Error occurred in MultidisciplinarySpatialTeam: {e}")
            return f"Error: {e!s}"


class AnnotationSpecialist(SpatialAnalysisAgent):
    """Annotation specialist agent."""

    def __init__(self, data_description: str, model_name: str = "gpt-4o"):
        """Initialize annotation specialist agent with data description and model."""
        super().__init__(data_description, "Annotation Specialist", model_name)


class CommunicationAnalyst(SpatialAnalysisAgent):
    """Communication analysis specialist agent."""

    def __init__(self, data_description: str, model_name: str = "gpt-4o"):
        """Initialize communication analyst agent with data description and model."""
        super().__init__(data_description, "Communication Analyst", model_name)


class DomainExpert(SpatialAnalysisAgent):
    """Spatial domain expert agent."""

    def __init__(self, data_description: str, model_name: str = "gpt-4o"):
        """Initialize domain expert agent with data description and model."""
        super().__init__(data_description, "Spatial Domain Expert", model_name)


async def run_spatial_analysis(data_description: str, model_name: str = "gpt-4o") -> str:
    """Run the complete spatial analysis pipeline.

    Args:
        data_description: Description of the spatial transcriptomics data
        model_name: The name of the model to use

    Returns:
        Final integrated analysis plan from the multidisciplinary team
    """
    # Create specialist agents with their own model instances
    agents = {
        "Annotation Specialist": AnnotationSpecialist(data_description, model_name),
        "Communication Analyst": CommunicationAnalyst(data_description, model_name),
        "Spatial Domain Expert": DomainExpert(data_description, model_name),
    }

    # Run all agents concurrently
    tasks = [agent.run() for agent in agents.values()]
    responses = await asyncio.gather(*tasks)

    # Map responses back to agent names
    agent_names = list(agents.keys())
    specialist_reports = dict(zip(agent_names, responses, strict=False))

    # Create and run multidisciplinary team
    team = MultidisciplinarySpatialTeam(
        annotation_report=specialist_reports["Annotation Specialist"],
        communication_report=specialist_reports["Communication Analyst"],
        domain_report=specialist_reports["Spatial Domain Expert"],
        model_name=model_name,
    )

    # Run the team and get final analysis plan
    final_plan = await team.run()
    return final_plan




