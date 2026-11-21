from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    sources: List[str]
    relevant: bool
    retries: int
    answer_supported: bool
    answer_retries: int
