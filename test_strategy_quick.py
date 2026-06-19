from dotenv import load_dotenv
load_dotenv()

import json
import re

import pandas as pd

from agents.planner_agent import PlannerAgent
from agents.strategy_agent import StrategyAgent

df = pd.read_csv("datasets/credit_data_scored.csv")

planner = PlannerAgent()
strategy = StrategyAgent(df)

queries = [
    "maximize profit",
    "reduce risk",
    "increase approvals",
    "increase approvals but keep risk moderate",
]

print(f"{'Query':<45} {'Threshold'}")
print("-" * 55)

for q in queries:
    obj = planner.plan(q)
    out = strategy.decide(obj)
    match = re.search(r'"threshold"\s*:\s*([\d.]+)', out)
    threshold = match.group(1) if match else "?"
    print(f"{q:<45} {threshold}")
