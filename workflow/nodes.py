import pandas as pd

from agents.planner_agent import PlannerAgent
from agents.strategy_agent import StrategyAgent
from agents.explanation_agent import ExplanationAgent

from workflow.state import CreditPolicyState

planner = PlannerAgent()
explanation_agent = ExplanationAgent()


def planner_node(state: CreditPolicyState) -> dict:
    objective = planner.plan(state["user_query"])
    print("\nPlanner Output")
    print(objective)
    return {"objective": objective}


def strategy_node(state: CreditPolicyState) -> dict:
    strategy_agent = StrategyAgent(state["df"])
    recommendation = strategy_agent.decide(
        state["objective"],
    )
    print("\nStrategy Output")
    print(recommendation)
    return {"recommendation": recommendation}


def explanation_node(state: CreditPolicyState) -> dict:
    explanation = explanation_agent.explain(
        state["recommendation"]
    )
    return {"explanation": explanation}
