from langgraph.graph import StateGraph
from agents.state import TutorState
from agents.rag_node import rag_retrieve_node
from agents.tutor_agent import tutor_agent
from agents.coach_agent import coach_agent
from agents.critic_agent import critic_agent
from agents.parliament import parliament_node
from core.llm_client import LLMClient
from analystics.feature_extractor import FeatureExtractorLLM
from core.llm_client import LLMClient
from affect.emotion_model import EmotionDetector
from analystics.risk_model import RiskModelLLM

llm = LLMClient()
fx = FeatureExtractorLLM(llm_client=llm, max_retries=2)
risk_model = RiskModelLLM(feature_extractor=fx)


def build_graph(memory):
    llm = LLMClient()
    emotion_detector = EmotionDetector()

    def affective_node(state):
        emotion = emotion_detector.detect(state["user_input"])
        return {"emotion": emotion}

    def risk_node(state):
        res = risk_model.predict(state)
        return {
            "risk_score": res.score,
            "risk_level": res.level,
            "risk_reasons": res.reasons,
        }

    graph = StateGraph(TutorState)

    graph.add_node("rag", lambda s: rag_retrieve_node(s, memory, k=6, depth=2))
    graph.add_node("affect", affective_node)

    graph.add_node("tutor", lambda s: tutor_agent(s, llm))
    graph.add_node("coach", lambda s: coach_agent(s, llm))
    graph.add_node("critic", lambda s: critic_agent(s, llm))

    graph.add_node("parliament", parliament_node)
    graph.add_node("risk", risk_node)

    graph.set_entry_point("rag")

    # RAG 先执行，再做情绪，再并行
    graph.add_edge("rag", "affect")

    graph.add_edge("affect", "tutor")
    graph.add_edge("affect", "coach")
    graph.add_edge("affect", "critic")

    graph.add_edge("tutor", "parliament")
    graph.add_edge("coach", "parliament")
    graph.add_edge("critic", "parliament")

    graph.add_edge("parliament", "risk")

    return graph.compile()
