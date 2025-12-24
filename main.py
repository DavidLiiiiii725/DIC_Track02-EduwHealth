from core.orchestrator import TutorOrchestrator

if __name__ == "__main__":
    tutor = TutorOrchestrator()

    while True:
        user_input = input("Student > ")
        if user_input.lower() in ["exit", "quit"]:
            break

        output = tutor.handle(user_input)

        print("\n[DEBUG] RAG Context:")
        print(output["rag_context"][:1200])
        print("\nAI Tutor Response:")
        print(output["response"])
        print("\nRisk Score:", output["risk"])
        print("Escalation:", output["escalation"])
        print("-" * 50)
