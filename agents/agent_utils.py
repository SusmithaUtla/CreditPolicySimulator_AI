from langchain_core.messages import AIMessage


def last_agent_message(result: dict) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return result["messages"][-1].content
