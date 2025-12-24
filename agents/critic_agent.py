def critic_agent(state, llm):
    rag = state.get("rag_context", "")

    user = f"""
Check safety and ethics risks. If user suggests self-harm, crisis, or severe distress, flag it clearly.
Use the retrieved context only as reference.

{rag}

User message:
{state["user_input"]}
""".strip()

    text = llm.chat(
        system="You are a safety and ethics monitor for an educational tutor.",
        user=user,
        temperature=0.2
    )
    return {"critic_response": text}
