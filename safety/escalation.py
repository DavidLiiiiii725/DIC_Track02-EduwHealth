class HumanEscalation:
    def check(self, risk_score):
        if risk_score > 0.8:
            return "ESCALATE_TO_HUMAN"
        return "OK"
