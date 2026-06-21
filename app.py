import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cloud: Streamlit secrets | Local: .env file
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    else:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

from workflow.langgraph_workflow import graph
from tools.simulation_tool import SimulationTool
from tools.risk_tool import RiskTool

PRESET_QUERIES = [
    "maximize profit",
    "reduce risk",
    "increase approvals",
    "increase approvals but keep risk moderate",
]


@st.cache_data
def load_portfolio_data() -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / "datasets/credit_data_scored.csv"
    )


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_workflow(user_query: str, df: pd.DataFrame) -> dict:
    return graph.invoke({
        "user_query": user_query,
        "df": df,
    })


def build_simulation_chart(df: pd.DataFrame) -> pd.DataFrame:
    simulations = SimulationTool().run(df)
    risks = [
        RiskTool().analyze(df, row["threshold"])
        for row in simulations
    ]
    chart_df = pd.DataFrame([
        {**sim, **risk}
        for sim, risk in zip(simulations, risks)
    ])
    return chart_df.set_index("threshold")


def main():
    st.set_page_config(
        page_title="Credit Policy Simulator",
        page_icon="📊",
        layout="wide",
    )

    st.title("Credit Policy Simulator AI")
    st.caption(
        "Multi-agent workflow: Planner → Strategy (with tools) → Explanation"
    )

    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY not found. Add it to your `.env` file in the project root."
        )
        st.stop()

    df = load_portfolio_data()

    with st.sidebar:
        st.header("Business goal")
        preset = st.selectbox(
            "Quick presets",
            ["Custom..."] + PRESET_QUERIES,
        )

        default_query = "" if preset == "Custom..." else preset
        user_query = st.text_area(
            "Describe your credit policy goal",
            value=default_query,
            height=100,
            placeholder="e.g. maximize profit while keeping risk low",
        )

        st.divider()
        st.subheader("Portfolio snapshot")
        st.metric("Customers", f"{len(df):,}")
        st.metric(
            "Avg default probability",
            f"{df['predicted_default_probability'].mean():.2%}",
        )
        st.metric(
            "Total loan volume",
            f"${df['loan_amount'].sum():,.0f}",
        )

        run_clicked = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
        )

    if not run_clicked and "result" not in st.session_state:
        st.info("Select or enter a business goal, then click **Run analysis**.")
        with st.expander("Preview: threshold simulation data"):
            st.dataframe(build_simulation_chart(df), use_container_width=True)
        return

    if run_clicked:
        if not user_query.strip():
            st.warning("Please enter a business goal first.")
            return

        try:
            with st.status("Running multi-agent workflow...", expanded=True) as status:
                st.write("Planner agent — interpreting your goal...")
                result = run_workflow(user_query.strip(), df)
                st.write("Strategy agent — calling simulation & risk tools...")
                st.write("Explanation agent — drafting executive summary...")
                status.update(
                    label="Analysis complete",
                    state="complete",
                )
            st.session_state["result"] = result
            st.session_state["user_query"] = user_query.strip()
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")
            return

    result = st.session_state.get("result")
    if not result:
        return

    st.subheader(f"Goal: {st.session_state.get('user_query', '')}")

    recommendation = extract_json(result.get("recommendation", ""))

    if recommendation:
        cols = st.columns(4)
        cols[0].metric(
            "Threshold",
            recommendation.get("threshold", "—"),
        )
        cols[1].metric(
            "Approval rate",
            f"{recommendation.get('approval_rate', 0)}%",
        )
        cols[2].metric(
            "Expected profit",
            f"${recommendation.get('profit', 0):,.0f}",
        )
        cols[3].metric(
            "Risk status",
            recommendation.get("risk_status", "—"),
        )

    tab_planner, tab_strategy, tab_summary, tab_data = st.tabs([
        "Planner",
        "Strategy",
        "Executive summary",
        "Simulation data",
    ])

    with tab_planner:
        st.markdown("**Structured objective from Planner agent**")
        objective_json = extract_json(result.get("objective", ""))
        if objective_json:
            st.json(objective_json)
        else:
            st.code(result.get("objective", ""))

    with tab_strategy:
        st.markdown("**Threshold recommendation from Strategy agent**")
        st.markdown(
            "_Strategy agent uses `run_credit_simulation` and "
            "`analyze_all_threshold_risks` tools via ReAct._"
        )
        if recommendation:
            st.json(recommendation)
        else:
            st.code(result.get("recommendation", ""))

    with tab_summary:
        st.markdown("**Final recommendation for executives**")
        st.markdown(result.get("explanation", ""))

    with tab_data:
        st.markdown("**Profit and risk by threshold**")
        chart_df = build_simulation_chart(df)
        st.line_chart(chart_df[["profit", "approval_rate"]])
        st.dataframe(chart_df, use_container_width=True)


if __name__ == "__main__":
    main()
