from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class ExplanationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )

    def explain(
        self,
        risk_summary,
        recommendation,
        scenarios
    ):

        prompt = f"""
        You are a Senior Credit Risk Analyst.

        Risk Summary:
        {risk_summary}

        Recommendation:
        {recommendation}

        Scenario Results:
        {scenarios}

        Explain:
        - Why this policy was selected
        - Risk/Profit tradeoff
        - Business recommendation
        """

        return self.llm.invoke(
            prompt
        ).content