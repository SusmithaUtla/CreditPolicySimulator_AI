import json

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.agent_utils import last_agent_message

EXPLANATION_PROMPT = """
You are a Credit Policy Explanation agent for business executives.

When given a recommendation, use the parse_recommendation tool to inspect the JSON fields,
then write a clear executive summary covering:
- chosen threshold and what it means
- approval rate and profit impact
- risk level and business tradeoff
- final actionable recommendation

Write in plain business language, not JSON.
"""


def create_explanation_tools(recommendation: str) -> list:

    @tool
    def parse_recommendation() -> str:
        """Parse and return the structured recommendation JSON for explanation."""
        return recommendation

    return [parse_recommendation]


class ExplanationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )

    def explain(self, recommendation: str) -> str:

        tools = create_explanation_tools(recommendation)

        agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=EXPLANATION_PROMPT,
            name="explanation_agent",
        )

        result = agent.invoke({
            "messages": [
                (
                    "user",
                    "Explain this credit policy recommendation to an executive.\n"
                    f"Recommendation JSON:\n{recommendation}",
                ),
            ],
        })

        return last_agent_message(result)
