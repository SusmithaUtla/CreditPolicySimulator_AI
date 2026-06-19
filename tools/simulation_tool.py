import pandas as pd


class SimulationTool:

    def run(self, df):

        results = []

        thresholds = [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8
        ]

        for threshold in thresholds:

            approved = df[
                df[
                    "predicted_default_probability"
                ] < threshold
            ]

            revenue = (
                approved["loan_amount"]
                * 0.15
            ).sum()

            expected_loss = (
                approved[
                    "predicted_default_probability"
                ]
                * approved["loan_amount"]
                * 0.30
            ).sum()

            profit = (
                revenue
                - expected_loss
            )

            results.append({
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
                        profit,
                        2
                    )
            })

        return results
