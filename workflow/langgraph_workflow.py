import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langgraph.graph import END, StateGraph

from workflow.state import CreditPolicyState
from workflow.nodes import (
    planner_node,
    strategy_node,
    explanation_node,
)

builder = StateGraph(
    CreditPolicyState
)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "strategy",
    strategy_node
)

builder.add_node(
    "explanation",
    explanation_node
)

builder.set_entry_point(
    "planner"
)

builder.add_edge(
    "planner",
    "strategy"
)

builder.add_edge(
    "strategy",
    "explanation"
)

builder.add_edge(
    "explanation",
    END
)

graph = builder.compile()


if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv(
        PROJECT_ROOT / "datasets/credit_data_scored.csv"
    )

    user_query = (
        "increase approvals but keep risk moderate"
    )

    result = graph.invoke({
        "user_query": user_query,
        "df": df,
    })

    print("\nFinal Recommendation")
    print("=" * 60)
    print(result["explanation"])
