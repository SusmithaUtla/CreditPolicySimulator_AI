class StrategyAgent:

    def recommend(
        self,
        scenarios
    ):

        best = max(
            scenarios,
            key=lambda x:
            x["profit"]
        )

        return {
            "recommended_threshold":
                best["threshold"],
            "expected_profit":
                best["profit"]
        }
