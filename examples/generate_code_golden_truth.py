"""Generate golden truth code solutions for DataBench QA using Claude Opus."""

import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from anthropic import Anthropic
from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Model options: haiku (cheap/fast), sonnet (balanced)
MODELS = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
}
CURRENT_MODEL = "sonnet"


def load_client():
    """Initialize and return Anthropic client."""
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(question: str, df: pd.DataFrame, expected_type: str) -> str:
    """Build prompt for code generation given a question and dataframe."""
    columns = list(df.columns)
    head_idx = list(range(min(2, len(df))))
    random_idx = df.sample(n=min(2, len(df)), random_state=42).index.tolist()
    sample_idx = list(dict.fromkeys(head_idx + random_idx))  # unique, preserves order
    sample_rows = df.iloc[sample_idx].to_string()

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


def build_error_prompt(question: str, df: pd.DataFrame, expected_type: str,
                       code: str, error: str) -> str:
    """Build prompt to fix code that failed to execute."""
    columns = list(df.columns)
    return f"""Your previous code failed with an error. Fix it.

Question: {question}
Columns: {columns}
Expected return type: {expected_type}

Your code:
{code}

Error:
{error}

Rules:
- Return ONLY the corrected code, no explanations
- The code should be a single expression
- Use only pandas/numpy operations

Fixed code:"""


def build_mismatch_prompt(question: str, df: pd.DataFrame, expected_type: str,
                          code: str, result, expected: str) -> str:
    """Build prompt to fix code that returned wrong answer."""
    columns = list(df.columns)
    # Truncate long results to avoid API errors
    result_str = str(result)[:200]
    expected_str = str(expected)[:200]
    return f"""Your code returned the wrong answer. Fix it.

Question: {question}
Columns: {columns}
Expected type: {expected_type}

Your code: {code}
Your result: {result_str}
Expected: {expected_str}

Return ONLY the corrected code:"""


def clean_code(code: str) -> str:
    """Remove markdown formatting from code response."""
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]  # remove first line
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def generate_code(client: Anthropic, prompt: str) -> str:
    """Call Claude to generate pandas code for the given prompt."""
    try:
        response = client.messages.create(
            model=MODELS[CURRENT_MODEL],
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_code(response.content[0].text)
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


def generate_with_retry(client, row, df, max_error_retries=3, max_mismatch_retries=2):
    """Generate code with auto-correction on errors and mismatches."""
    prompt = build_prompt(row["question"], df, row["type"])
    code = generate_code(client, prompt)
    attempts = 1

    # Phase 1: Retry on execution errors
    for _ in range(max_error_retries):
        result = execute_code(code, df)
        if not str(result).startswith("__EXEC_ERROR__"):
            break
        error_prompt = build_error_prompt(
            row["question"], df, row["type"], code, str(result)
        )
        code = generate_code(client, error_prompt)
        attempts += 1

    # Check if still erroring after retries
    if str(result).startswith("__EXEC_ERROR__"):
        return code, result, attempts

    # Phase 2: Retry on result mismatch
    for _ in range(max_mismatch_retries):
        if verify_result(result, row["answer"], row["type"]):
            return code, result, attempts
        mismatch_prompt = build_mismatch_prompt(
            row["question"], df, row["type"], code, result, row["answer"]
        )
        code = generate_code(client, mismatch_prompt)
        attempts += 1
        result = execute_code(code, df)
        if str(result).startswith("__EXEC_ERROR__"):
            break  # Don't keep retrying if new code errors

    return code, result, attempts


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
        code, result, attempts = generate_with_retry(client, row, df)
        verified = verify_result(result, row["answer"], row["type"])

        results.append({
            "question": row["question"],
            "dataset": row["dataset"],
            "answer": row["answer"],
            "type": row["type"],
            "code_solution": code,
            "code_verified": verified,
            "attempts": attempts
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code solutions for DataBench QA")
    parser.add_argument("--lang", choices=["EN", "ES", "both"], default="both")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="golden_truth")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="sonnet",
                        help="Model to use: haiku (cheap) or sonnet (better)")
    args = parser.parse_args()

    CURRENT_MODEL = args.model  # noqa: F841
    print(f"Using model: {MODELS[args.model]}")
    langs = ["EN", "ES"] if args.lang == "both" else [args.lang]

    for lang in langs:
        print(f"\nProcessing {lang} dataset...")
        df = process_dataset(lang=lang, split=args.split, limit=args.limit)
        output_path = f"{args.output}_{lang.lower()}.csv"
        df.to_csv(output_path, index=False)
        verified_count = df["code_verified"].sum()
        print(f"Saved to {output_path} - {verified_count}/{len(df)} verified")
