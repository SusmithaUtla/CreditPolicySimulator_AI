import json

import pandas as pd
from langchain_core.tools import tool

from tools.simulation_tool import SimulationTool
from tools.risk_tool import RiskTool


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def create_credit_tools(df: pd.DataFrame) -> list:
    simulation_tool = SimulationTool()
    risk_tool = RiskTool()

    @tool
    def run_credit_simulation() -> str:
        """Run profit and approval-rate simulations across credit policy thresholds (0.1 to 0.8)."""
        results = simulation_tool.run(df)
        return json.dumps(results)

    @tool
    def analyze_portfolio_risk(threshold: float) -> str:
        """Analyze average default risk and risk status (LOW/MEDIUM/HIGH) at one approval threshold."""
        result = risk_tool.analyze(df, threshold)
        return json.dumps(result)

    @tool
    def analyze_all_threshold_risks() -> str:
        """Analyze risk metrics for every policy threshold from 0.1 through 0.8."""
        results = [
            risk_tool.analyze(df, threshold)
            for threshold in THRESHOLDS
        ]
        return json.dumps(results)

    return [
        run_credit_simulation,
        analyze_portfolio_risk,
        analyze_all_threshold_risks,
    ]
