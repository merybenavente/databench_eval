"""Generate golden truth code solutions for DataBench QA using Claude Opus."""

import os
import pandas as pd
import numpy as np

from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def load_client():
    """Initialize and return Anthropic client."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def build_prompt(question: str, df: pd.DataFrame, expected_type: str) -> str:
    """Build prompt for code generation given a question and dataframe."""
    columns = list(df.columns)
    head_rows = df.head(2)
    random_rows = df.sample(n=min(2, len(df)), random_state=42)
    sample_rows = pd.concat([head_rows, random_rows]).drop_duplicates().to_string()

    return f"""You are a Principal Data Scientiet and pandas expert. Generate a single line of Python code that answers the question below.

The dataframe `df` has these columns: {columns}

Sample rows:
{sample_rows}

Question: {question}
Expected return type: {expected_type}

Rules:
- Return ONLY the code, no explanations
- The code should be a single expression (no variable assignments)
- Use only pandas/numpy operations
- For boolean: return True or False (not np.bool_)
- For lists: return a Python list

Code:"""


def generate_code(client: Anthropic, prompt: str) -> str:
    """Call Claude to generate pandas code for the given prompt."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"__ERROR__: {e}"
