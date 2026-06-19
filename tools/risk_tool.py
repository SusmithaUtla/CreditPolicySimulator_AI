class RiskTool:

    def analyze(
        self,
        df,
        threshold
    ):

        approved = df[
            df[
                "predicted_default_probability"
            ] < threshold
        ]

        avg_risk = approved[
            "predicted_default_probability"
        ].mean()

        if avg_risk < 0.2:
            risk_status = "LOW"

        elif avg_risk < 0.35:
            risk_status = "MEDIUM"

        else:
            risk_status = "HIGH"

        return {
            "threshold": threshold,
            "avg_risk":
                round(avg_risk, 4),
            "risk_status":
                risk_status
        }
