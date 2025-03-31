from abc import ABC, abstractmethod
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_model, move_to_device
from .agent_name import AgentName

class BaseAgent(ABC):
    """Base agent class that all other agents will inherit from."""
    
    def __init__(self, model=None):
        """Initialize with either a provided model or load the default one."""
        if model is None:
            self.model, self.device = move_to_device(load_model())
        else:
            self.model, self.device = move_to_device(model)
    
    @abstractmethod
    def run(self, input_text):
        """Run the agent on the provided input."""
        pass


def create_agent(agent_name, model=None):
    """
    Factory function to create the appropriate agent based on the agent name.
    
    Args:
        agent_name (AgentName): The type of agent to create
        model: Optional pre-loaded model to use
        
    Returns:
        BaseAgent: An instance of the requested agent type
    """
    if agent_name == AgentName.QA:
        from .qa_agent import QAAgent
        return QAAgent(model)
    elif agent_name == AgentName.CHAT:
        from .chat_agent import ChatAgent
        return ChatAgent(model)
    elif agent_name == AgentName.TOOL:
        from .tool_agent import ToolAgent
        return ToolAgent(model)
    elif agent_name == AgentName.NAIVE:
        from .naive_agent import NaiveAgent
        return NaiveAgent(model)
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")
