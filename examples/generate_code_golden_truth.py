"""Generate golden truth code solutions for DataBench QA using Claude Opus."""

import os
import pandas as pd
import numpy as np

from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def load_client():
    """Initialize and return Anthropic client."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)
