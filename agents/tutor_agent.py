def tutor_agent(state, llm):
    rag = state.get("rag_context", "")

    user = f"""
You MUST use the following retrieved knowledge as your primary grounding.
If the knowledge is insufficient, say what is missing and ask one clarifying question.

{rag}

Student question:
{state["user_input"]}
""".strip()

    text = llm.chat(
        system="You are an academic tutor. Be precise, structured, and grounded in retrieved context.",
        user=user,
        temperature=0.4
    )
    return {"tutor_response": text}
