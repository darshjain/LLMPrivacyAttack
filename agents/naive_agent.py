# agents/naive_agent.py
from .base_agent import BaseAgent
from transformers import pipeline

class NaiveAgent(BaseAgent):
    """
    A simple naive agent that directly uses the model without LangChain.
    This maintains compatibility with existing code.
    """
    
    def __init__(self, model=None):
        super().__init__(model)
        
        # Create a text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            device=self.device.index if hasattr(self.device, 'index') else -1
        )
    
    def run(self, input_text):
        """Run the agent on the provided input."""
        outputs = self.pipeline(
            input_text,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Return the generated text
        return outputs[0]['generated_text']
