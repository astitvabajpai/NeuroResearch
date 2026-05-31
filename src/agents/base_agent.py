from abc import ABC, abstractmethod
from src.state.research_state import ResearchState

class BaseAgent(ABC):
    @abstractmethod
    def invoke(self, state: ResearchState) -> dict:
        pass