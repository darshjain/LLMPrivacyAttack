from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import HuggingFaceTextGenInference

from .base_agent import BaseAgent

class ToolAgent(BaseAgent):
    """Agent that can use tools to accomplish tasks."""
    
    def __init__(self, model=None):
        super().__init__(model)
        
        self.llm = HuggingFaceTextGenInference(
            inference_server_url="http://localhost:8080/generate",  
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.3,
            repetition_penalty=1.03
        )
        
        self.tools = [
            Tool(
                name="Search",
                func=DuckDuckGoSearchRun().run,
                description="Useful for when you need to answer questions about current events or the world"
            )
        ]
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def run(self, task):
        """Run the agent on the provided task."""
        return self.agent.run(task)
