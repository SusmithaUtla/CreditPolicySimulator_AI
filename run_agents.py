import pandas as pd

from agents.risk_agent import RiskAgent
from agents.simulation_agent import SimulationAgent
from agents.strategy_agent import StrategyAgent
from agents.explanation_agent import ExplanationAgent

df = pd.read_csv(
    "datasets/credit_data_scored.csv"
)

risk_agent = RiskAgent()
simulation_agent = SimulationAgent()
strategy_agent = StrategyAgent()
explanation_agent = ExplanationAgent()

# ------------------------
# Risk Analysis
# ------------------------

risk_summary = risk_agent.analyze(df)

print("\nRisk Summary")
print(risk_summary)

# ------------------------
# Run Simulations
# ------------------------

scenarios = []

for threshold in [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
]:

    result = simulation_agent.run(
        df,
        threshold
    )

    scenarios.append(result)

# ------------------------
# Strategy Recommendation
# ------------------------

recommendation = (
    strategy_agent.recommend(
        scenarios
    )
)

print("\nRecommendation")
print(recommendation)

# ------------------------
# LLM Explanation
# ------------------------

try:
    analysis = explanation_agent.explain(
        risk_summary,
        recommendation,
        scenarios
    )

    print("\nLLM Analysis")
    print(analysis)

except Exception as e:
    print("ERROR:")
    print(e)