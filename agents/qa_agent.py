from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceTextGenInference

from .base_agent import BaseAgent

class QAAgent(BaseAgent):
    """Agent for answering questions using the loaded model."""
    
    def __init__(self, model=None):
        super().__init__(model)
        
        self.llm = HuggingFaceTextGenInference(
            inference_server_url="http://localhost:8080/generate", 
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.1,
            repetition_penalty=1.03
        )
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer the following question:\nQuestion: {question}\nAnswer:"
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def run(self, question):
        """Run the QA chain on the provided question."""
        return self.chain.run(question=question)
