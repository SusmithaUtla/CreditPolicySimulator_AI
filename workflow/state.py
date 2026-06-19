from typing import Any, TypedDict

import pandas as pd


class CreditPolicyState(TypedDict, total=False):
    user_query: str
    df: pd.DataFrame
    objective: str
    recommendation: str
    explanation: str
