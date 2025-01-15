"""Generate golden truth code solutions for DataBench QA using Claude Opus."""

import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from anthropic import Anthropic
from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table

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


def execute_code(code: str, df: pd.DataFrame):
    """Execute generated code against dataframe and return result."""
    try:
        local_vars = {"df": df, "pd": pd, "np": np}
        result = eval(code, {"__builtins__": {}}, local_vars)
        if isinstance(result, pd.Series):
            result = result.tolist()
        elif isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0].tolist()
        return result
    except Exception as e:
        return f"__EXEC_ERROR__: {e}"


def verify_result(result, expected: str, semantic_type: str) -> bool:
    """Check if the executed result matches the expected answer."""
    if isinstance(result, str) and result.startswith("__"):
        return False
    evaluator = Evaluator.__new__(Evaluator)
    return evaluator.default_compare(str(result), expected, semantic_type)


def process_dataset(lang: str, split: str = "dev", limit: int = None) -> pd.DataFrame:
    """Process QA dataset and generate code solutions."""
    name = "iberlef" if lang == "ES" else "semeval"
    qa = load_qa(lang=lang, name=name, split=split)

    if limit:
        qa = qa.select(range(min(limit, len(qa))))

    client = load_client()
    results = []

    for row in tqdm(qa, desc=f"Processing {lang}"):
        df = load_table(row["dataset"], lang=lang)
        prompt = build_prompt(row["question"], df, row["type"])
        code = generate_code(client, prompt)
        result = execute_code(code, df)
        verified = verify_result(result, row["answer"], row["type"])

        results.append({
            "question": row["question"],
            "dataset": row["dataset"],
            "answer": row["answer"],
            "type": row["type"],
            "code_solution": code,
            "code_verified": verified
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code solutions for DataBench QA")
    parser.add_argument("--lang", choices=["EN", "ES", "both"], default="both")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="golden_truth")
    args = parser.parse_args()

    langs = ["EN", "ES"] if args.lang == "both" else [args.lang]

    for lang in langs:
        print(f"\nProcessing {lang} dataset...")
        df = process_dataset(lang=lang, split=args.split, limit=args.limit)
        output_path = f"{args.output}_{lang.lower()}.csv"
        df.to_csv(output_path, index=False)
        verified_count = df["code_verified"].sum()
        print(f"Saved to {output_path} - {verified_count}/{len(df)} verified")
