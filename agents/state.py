from typing import TypedDict, Dict, NotRequired, List, Any

class TutorState(TypedDict):
    user_input: str
    rag_context: NotRequired[str]
    rag_evidence: NotRequired[Dict[str, Any]]
    rag_semantic: NotRequired[List[str]]
    rag_structured: NotRequired[List[str]]

    risk_level: NotRequired[str]
    risk_reasons: NotRequired[dict]

    # affect
    emotion: NotRequired[Dict[str, float]]

    # agents
    tutor_response: NotRequired[str]
    coach_response: NotRequired[str]
    critic_response: NotRequired[str]

    # output
    final_response: NotRequired[str]
    risk_score: NotRequired[float]

