def coach_agent(state, llm):
    rag = state.get("rag_context", "")

    user = f"""
You are a motivational coach grounded in retrieved knowledge.
Use supportive, autonomy-supportive language (SDT). Avoid medical claims.

{rag}

Student message:
{state["user_input"]}
""".strip()

    text = llm.chat(
        system="You are an empathetic motivational coach. Support autonomy, competence, and relatedness.",
        user=user,
        temperature=0.7
    )
    return {"coach_response": text}
