# analytics/feature_extractor_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import math


@dataclass
class ExtractedFeatures:
    """
    LLM-extracted features for risk scoring.
    All values should be numeric in [0,1] (except length_norm which is [0,1] too).
    """
    sadness: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    joy: float = 0.0

    # content-based risk indicators (LLM judged)
    self_harm_risk: float = 0.0       # 0..1: any self-harm / suicide ideation risk signals
    hopelessness: float = 0.0         # 0..1
    overwhelm: float = 0.0            # 0..1
    panic: float = 0.0               # 0..1
    functional_impairment: float = 0.0  # 0..1: sleep/eat/focus disruption

    # meta
    urgency: float = 0.0             # 0..1: how urgent is situation
    intensity: float = 0.0           # 0..1: emotional intensity
    negation_or_denial: float = 0.0  # 0..1: explicitly denies self-harm intent

    # context quality
    rag_empty: float = 0.0           # 0/1
    rag_len_norm: float = 0.0        # 0..1
    user_len_norm: float = 0.0       # 0..1

    def to_dict(self) -> Dict[str, float]:
        return {
            "sadness": self.sadness,
            "fear": self.fear,
            "anger": self.anger,
            "joy": self.joy,
            "self_harm_risk": self.self_harm_risk,
            "hopelessness": self.hopelessness,
            "overwhelm": self.overwhelm,
            "panic": self.panic,
            "functional_impairment": self.functional_impairment,
            "urgency": self.urgency,
            "intensity": self.intensity,
            "negation_or_denial": self.negation_or_denial,
            "rag_empty": self.rag_empty,
            "rag_len_norm": self.rag_len_norm,
            "user_len_norm": self.user_len_norm,
        }


class FeatureExtractorLLM:
    """
    Uses a local LLM to extract risk-related features WITHOUT manual keyword lists.

    Requirements:
      - llm_client must implement: chat(system: str, user: str, temperature: float=...) -> str
        (Your existing core.llm_client.LLMClient likely matches this.)
    """

    def __init__(self, llm_client, *, max_retries: int = 2):
        self.llm = llm_client
        self.max_retries = max_retries

    def extract(self, state: Dict[str, Any]) -> ExtractedFeatures:
        user_input = (state.get("user_input") or "").strip()
        rag_context = (state.get("rag_context") or "").strip()

        # Pre-compute simple non-lexical meta (allowed; not “keyword marking”)
        user_len_norm = self._length_norm(user_input)
        rag_len_norm = self._length_norm(rag_context)
        rag_empty = 1.0 if not rag_context else 0.0

        # Build LLM prompt
        system = (
            "You are a strict feature extractor for an educational tutor safety system.\n"
            "Task: Given a student's message, output ONLY valid JSON with numeric fields in [0,1].\n"
            "Do not output any other text.\n"
            "Do not diagnose; only estimate risk indicators.\n"
        )

        schema = {
            "sadness": "float 0..1",
            "fear": "float 0..1",
            "anger": "float 0..1",
            "joy": "float 0..1",
            "self_harm_risk": "float 0..1 (signals of self-harm or suicidal ideation, if any)",
            "hopelessness": "float 0..1",
            "overwhelm": "float 0..1",
            "panic": "float 0..1",
            "functional_impairment": "float 0..1 (sleep/eat/focus disruption, inability to function)",
            "urgency": "float 0..1 (how urgent it sounds to seek support)",
            "intensity": "float 0..1 (emotional intensity)",
            "negation_or_denial": "float 0..1 (explicitly denies self-harm intent)",
        }

        user = (
            "Return JSON with the following keys and meanings:\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            "Student message:\n"
            f"{user_input}\n"
        )

        # Ask LLM, parse JSON robustly
        raw = self._call_with_retries(system, user)

        data = self._safe_json_load(raw)

        # Map into dataclass with clamps + defaults
        feats = ExtractedFeatures(
            sadness=self._clamp01(data.get("sadness", 0.0)),
            fear=self._clamp01(data.get("fear", 0.0)),
            anger=self._clamp01(data.get("anger", 0.0)),
            joy=self._clamp01(data.get("joy", 0.0)),
            self_harm_risk=self._clamp01(data.get("self_harm_risk", 0.0)),
            hopelessness=self._clamp01(data.get("hopelessness", 0.0)),
            overwhelm=self._clamp01(data.get("overwhelm", 0.0)),
            panic=self._clamp01(data.get("panic", 0.0)),
            functional_impairment=self._clamp01(data.get("functional_impairment", 0.0)),
            urgency=self._clamp01(data.get("urgency", 0.0)),
            intensity=self._clamp01(data.get("intensity", 0.0)),
            negation_or_denial=self._clamp01(data.get("negation_or_denial", 0.0)),
            rag_empty=rag_empty,
            rag_len_norm=rag_len_norm,
            user_len_norm=user_len_norm,
        )

        # Optional safety tempering: if model both flags self-harm risk and strong denial,
        # reduce but do not zero.
        if feats.self_harm_risk > 0.6 and feats.negation_or_denial > 0.7:
            feats.self_harm_risk = max(0.2, feats.self_harm_risk * 0.5)

        return feats

    def _call_with_retries(self, system: str, user: str) -> str:
        last = ""
        for i in range(self.max_retries + 1):
            out = self.llm.chat(system=system, user=user, temperature=0.0)
            last = out.strip()
            # quick check: must start with "{" and end with "}"
            if last.startswith("{") and last.rstrip().endswith("}"):
                return last
            # retry with stricter instruction
            user = (
                "IMPORTANT: Output ONLY JSON. No markdown, no explanation.\n"
                + user
            )
        return last

    @staticmethod
    def _safe_json_load(text: str) -> Dict[str, Any]:
        """
        Attempt to parse JSON even if model adds stray text.
        """
        text = text.strip()
        # try direct
        try:
            return json.loads(text)
        except Exception:
            pass

        # salvage: extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v != v:  # NaN
            return 0.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _length_norm(text: str) -> float:
        n = max(0, len(text or ""))
        return float(min(1.0, math.log1p(n) / math.log1p(2000)))
