class SimulationAgent:

    def run(
        self,
        df,
        threshold,
        interest_rate=0.15,
        lgd=0.30
    ):

        approved = df[
            df[
              "predicted_default_probability"
            ] < threshold
        ]

        revenue = (
            approved["loan_amount"]
            * interest_rate
        ).sum()

        expected_loss = (
            approved[
              "predicted_default_probability"
            ]
            * approved["loan_amount"]
            * lgd
        ).sum()

        return {
            "threshold": threshold,
            "approval_rate":
                round(
                    len(approved)
                    / len(df)
                    * 100,
                    2
                ),
            "profit":
                round(
                    revenue - expected_loss,
                    2
                )
        }
