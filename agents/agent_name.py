from enum import Enum

class AgentName(Enum):
    QA = "qa"           # Question-answering agent
    CHAT = "chat"       # Conversational chat agent
    TOOL = "tool"       # Tool-using agent
    NAIVE = "naive"     
