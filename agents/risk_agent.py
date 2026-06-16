import pandas as pd


class RiskAgent:

    def analyze(self, df):

        return {
            "avg_risk":float(
                round(
                    df[
                        "predicted_default_probability"
                    ].mean(),
                    4
                )),

            "high_risk_customers":
                len(
                    df[
                        df[
                          "predicted_default_probability"
                        ] > 0.6
                    ]
                ),

            "portfolio_size":
                len(df)
        }
