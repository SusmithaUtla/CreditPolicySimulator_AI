from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.agent_utils import last_agent_message

PLANNER_PROMPT = """
You are a Credit Policy Planner agent.

Convert the user's business goal into a JSON objective.

Rules:
- maximize profit -> {"objective": "profit"}
- reduce risk -> {"objective": "risk"}
- increase approvals -> {"objective": "approval_rate"}
- compound goals (e.g. increase approvals but keep risk moderate) -> {"objectives": [{"objective": "approval_rate"}, {"objective": "risk", "target": "moderate"}]}

Return JSON only in your final answer. No markdown fences.
"""


class PlannerAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=[],
            prompt=PLANNER_PROMPT,
            name="planner_agent",
        )

    def plan(self, user_query: str) -> str:

        result = self.agent.invoke({
            "messages": [
                ("user", user_query),
            ],
        })

        return last_agent_message(result)
