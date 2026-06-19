import json
import re

import pandas as pd
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.agent_utils import last_agent_message
from tools.langchain_tools import create_credit_tools

STRATEGY_PROMPT = """
You are a Credit Policy Strategy agent.

You MUST use your tools before making a decision:
1. Call run_credit_simulation to get profit and approval metrics.
2. Call analyze_all_threshold_risks to get risk metrics.

Then pick exactly ONE threshold based on the user's objective:

- objective "profit" -> highest profit
- objective "risk" -> lowest avg_risk
- objective "approval_rate" -> highest approval_rate
- multiple objectives -> highest approval_rate where risk_status is MEDIUM

Return JSON only in your final answer:
{"threshold": 0.5, "approval_rate": 56.76, "profit": 292684229.22, "avg_risk": 0.2141, "risk_status": "MEDIUM"}
"""


class StrategyAgent:

    def __init__(self, df: pd.DataFrame | None = None):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )

        self._df = df
        self._agent = None

        if df is not None:
            self._build_agent(df)

    def _build_agent(self, df: pd.DataFrame):

        self._df = df
        tools = create_credit_tools(df)

        self._agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=STRATEGY_PROMPT,
            name="strategy_agent",
        )

    def _merge_results(self, simulations, risks):
        risk_by_threshold = {
            r["threshold"]: r for r in risks
        }
        return [
            {**sim, **risk_by_threshold[sim["threshold"]]}
            for sim in simulations
        ]

    def _parse_objective(self, objective):
        text = objective.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return text

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return text

        if "objectives" in data:
            parts = [
                o.get("objective", o.get("name", ""))
                for o in data["objectives"]
            ]
            return " ".join(parts)

        if "objective" in data:
            obj = data["objective"]
            if "constraint" in data:
                constraint = data["constraint"]
                if isinstance(constraint, dict):
                    extra = constraint.get(
                        "objective",
                        constraint.get("metric", "risk"),
                    )
                    return f"{obj} {extra}"
            if "constraints" in data:
                extras = [
                    c.get("objective", c.get("metric", "risk"))
                    for c in data["constraints"]
                ]
                return f"{obj} {' '.join(extras)}"
            return obj

        return text

    def _pick_by_objective(self, objective, rows):
        obj = objective.strip().lower()

        # compound: approval + risk constraint -> best approval among MEDIUM
        if "approval" in obj and "risk" in obj:
            medium = [
                r for r in rows
                if r["risk_status"] == "MEDIUM"
            ]
            pool = medium if medium else rows
            return max(
                pool,
                key=lambda r: r["approval_rate"],
            )

        if "profit" in obj:
            return max(rows, key=lambda r: r["profit"])

        if "risk" in obj:
            return min(rows, key=lambda r: r["avg_risk"])

        if "approval" in obj:
            return max(
                rows,
                key=lambda r: r["approval_rate"],
            )

        return max(rows, key=lambda r: r["profit"])

    def _run_tools(self, df: pd.DataFrame):
        from tools.simulation_tool import SimulationTool
        from tools.risk_tool import RiskTool

        simulation_tool = SimulationTool()
        risk_tool = RiskTool()

        simulations = simulation_tool.run(df)
        risks = [
            risk_tool.analyze(df, row["threshold"])
            for row in simulations
        ]
        return simulations, risks

    def decide_rules(
        self,
        objective,
        simulations,
        risks,
    ):
        rows = self._merge_results(simulations, risks)
        parsed = self._parse_objective(objective)
        chosen = self._pick_by_objective(parsed, rows)

        return json.dumps({
            "threshold": chosen["threshold"],
            "approval_rate": chosen["approval_rate"],
            "profit": round(float(chosen["profit"]), 2),
            "avg_risk": chosen["avg_risk"],
            "risk_status": chosen["risk_status"],
        })

    def decide(self, objective: str, df: pd.DataFrame | None = None) -> str:
        """Run simulation + risk tools, then apply rule-based threshold selection."""

        if df is not None:
            self._df = df
        elif self._df is None:
            raise ValueError(
                "StrategyAgent requires df. "
                "Pass df to __init__ or decide()."
            )

        simulations, risks = self._run_tools(self._df)
        return self.decide_rules(objective, simulations, risks)

    def decide_react(self, objective: str, df: pd.DataFrame | None = None) -> str:
        """Optional LLM-driven strategy (non-deterministic)."""

        if df is not None:
            self._build_agent(df)
        elif self._agent is None:
            raise ValueError(
                "StrategyAgent requires df. "
                "Pass df to __init__ or decide_react()."
            )

        result = self._agent.invoke({
            "messages": [
                (
                    "user",
                    f"Objective:\n{objective}\n\n"
                    "Use tools, then return the best threshold as JSON.",
                ),
            ],
        })

        return last_agent_message(result)
