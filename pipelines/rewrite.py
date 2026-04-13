# rewrite.py
"""
Clinical case rewriting module for pipeline.

This script exposes a function `rewrite_case_from_txt` that takes a TXT file
with the patient case and a model name, and returns a rewritten case in
guideline-style format.
The fuction 'translate_case_from_txt' takes a TXT file
with the patient case and a model name, and returns a translated only case in
english.
"""

import os
import sys

# -----------------------------
# Add the 'prompts' folder to sys.path
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.abspath(os.path.join(current_dir, '..', 'prompts'))
sys.path.append(prompts_dir)

# -----------------------------
# Imports
# -----------------------------
from prompt_templates import REWRITING_PROMPT, TRANSLATING_PROMPT
from chatgpt import chatgpt_chat_completion

# -----------------------------
# Main function
# -----------------------------
def rewrite_case_from_txt(txt_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Rewrites a patient case from a TXT file using a guideline-style prompt.

    Args:
        txt_path: path to the TXT file containing the original patient case.
        model: model name to use for ChatCompletion (default: "gpt-4o-mini").

    Returns:
        Rewritten patient case as a string.
    """
    # Load original case
    with open(txt_path, 'r', encoding='utf-8') as f:
        original_case = f.read().strip()

    # Format prompt
    formatted_prompt = REWRITING_PROMPT.format(original_case=original_case)

    # Run model
    rewritten_case = chatgpt_chat_completion(formatted_prompt, model=model)

    return rewritten_case


def translate_case_from_txt(txt_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Translates a patient case from a TXT file using a guideline-style prompt.

    Args:
        txt_path: path to the TXT file containing the original patient case.
        model: model name to use for ChatCompletion (default: "gpt-4o-mini").

    Returns:
        Rewritten patient case as a string.
    """
    # Load original case
    with open(txt_path, 'r', encoding='utf-8') as f:
        original_case = f.read().strip()

    # Format prompt
    formatted_prompt = TRANSLATING_PROMPT.format(original_case=original_case)

    # Run model
    rewritten_case = chatgpt_chat_completion(formatted_prompt, model=model)

    return rewritten_case

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    case_txt_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_patients', 'example_case_de.txt'))
    # rewritten = rewrite_case_from_txt(case_txt_path)
    # print("=== Rewritten Case ===")
    # print(rewritten)

    #translated = translate_case_from_txt(case_txt_path)
    #print("=== Translated Only Case ===")
    #print(translated)
