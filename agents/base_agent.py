# agents/base_agent.py
from abc import ABC, abstractmethod
from langchain.llms.base import BaseLLM
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_model, move_to_device

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
