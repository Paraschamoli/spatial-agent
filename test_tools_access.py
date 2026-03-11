#!/usr/bin/env python3
"""Test script to verify agent can access all tools."""

import sys
import os

# Add the spatial_agent package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tools_import():
    """Test that all tools can be imported."""
    print("Testing tools import...")
    
    try:
        # Test importing tools module
        import spatial_agent.tools
        print("✅ Successfully imported spatial_agent.tools")
        
        # Test importing specific tool categories
        from spatial_agent.tools.databases import search_panglao
        print("✅ Successfully imported database tools")
        
        from spatial_agent.tools.analytics import preprocess_spatial_data
        print("✅ Successfully imported analytics tools")
        
        from spatial_agent.tools.literature import query_pubmed
        print("✅ Successfully imported literature tools")
        
        from spatial_agent.tools.interpretation import annotate_cell_types
        print("✅ Successfully imported interpretation tools")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing tools: {e}")
        return False

def test_agent_import():
    """Test that agent module can be imported."""
    print("\nTesting agent import...")
    
    try:
        from spatial_agent.agent import SpatialAgent, make_llm
        print("✅ Successfully imported SpatialAgent and make_llm")
        
        from spatial_agent.agent.utils import load_all_tools
        print("✅ Successfully imported load_all_tools")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing agent: {e}")
        return False

def test_tools_loading():
    """Test that tools can be loaded."""
    print("\nTesting tools loading...")
    
    try:
        from spatial_agent.agent.utils import load_all_tools
        
        # Try to load tools (this may fail due to missing dependencies)
        tools = load_all_tools(save_path="./test_experiments", data_path="./test_data")
        print(f"✅ Successfully loaded {len(tools)} tools")
        
        # Print first few tool names
        if tools:
            print(f"   First 5 tools: {[tool.name for tool in tools[:5]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading tools: {e}")
        return False

def test_skills_loading():
    """Test that skills can be loaded."""
    print("\nTesting skills loading...")
    
    try:
        from spatial_agent.agent.skills import SkillManager
        
        skills_dir = os.path.join(os.path.dirname(__file__), "spatial_agent", "skills")
        if os.path.exists(skills_dir):
            skill_manager = SkillManager(skills_dir)
            skills = skill_manager.load_skills()
            print(f"✅ Successfully loaded {len(skills)} skills")
            
            if skills:
                print(f"   Skills: {list(skills.keys())}")
            
            return True
        else:
            print(f"❌ Skills directory not found: {skills_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading skills: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing SpatialAgent Tools Access ===\n")
    
    results = []
    results.append(test_tools_import())
    results.append(test_agent_import())
    results.append(test_tools_loading())
    results.append(test_skills_loading())
    
    print(f"\n=== Summary ===")
    print(f"Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("🎉 All tests passed! Agent can access tools and skills.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    sys.exit(0 if all(results) else 1)
