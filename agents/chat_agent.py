from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFaceTextGenInference

from .base_agent import BaseAgent

class ChatAgent(BaseAgent):
    """Agent for having a conversation using the loaded model."""
    
    def __init__(self, model=None):
        super().__init__(model)
        
        self.llm = HuggingFaceTextGenInference(
            inference_server_url="http://localhost:8080/generate", 
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.7,  
            repetition_penalty=1.03
        )
        
        self.memory = ConversationBufferMemory()
        
        self.chain = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=True
        )
    
    def run(self, user_input):
        """Continue the conversation with the provided user input."""
        return self.chain.predict(input=user_input)
