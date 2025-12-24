def parliament_node(state):
    return {
        "final_response": f"""
[TUTOR – Competence]
{state.get('tutor_response', '')}

[COACH – Relatedness]
{state.get('coach_response', '')}

[CRITIC – Safety]
{state.get('critic_response', '')}
""".strip()
    }
