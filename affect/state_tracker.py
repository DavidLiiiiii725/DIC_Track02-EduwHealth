class EmotionalState:
    def __init__(self):
        self.history = []

    def update(self, emotion_scores):
        self.history.append(emotion_scores)

    def is_distressed(self):
        if not self.history:
            return False
        last = self.history[-1]
        return last.get("sadness", 0) > 0.4 or last.get("fear", 0) > 0.4
