# pipelines/simple_request_pipeline.py

"""
Run a SINGLE REQUEST prompt on a patient case (original or rewritten).

User selects the model at runtime. The case can be original or rewritten
(using the rewrite.py pipeline).
"""

import sys
import os

# -----------------------------
# Add the parallel 'prompts' folder to sys.path
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.abspath(os.path.join(current_dir, '..', 'prompts'))
sys.path.append(prompts_dir)

# -----------------------------
# Imports
# -----------------------------
from prompt_templates import get_prompt_for_configuration
from chatgpt import chatgpt_chat_completion

# Optionally import the rewrite function
# (Assuming rewrite.py exposes rewrite_case_from_txt function)
sys.path.append(os.path.abspath(os.path.join(current_dir)))


# -----------------------------
# User configuration
# -----------------------------
print("Select model (1 or 2):")
print("1) gpt-4o-mini")
print("2) gpt-4o")
model_choice = input("Enter choice [1 or 2]: ").strip()
MODEL_NAME = "gpt-4o-mini-2024-07-18" if model_choice == "1" else "gpt-4o-2024-11-20" 

use_rewritten = input("Use rewritten or translated only case? (y/n): ").strip().lower() == "y"

if use_rewritten:
    use_translated_only = input("Do you want to use the translated (tr) only case or the rewritten (rw) one? (tr/rw): ").strip().lower() == "tr"
else:
    use_translated_only = False

# -----------------------------
# Load original patient case
# -----------------------------
case_txt_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_patients', 'example_case_de.txt'))

with open(case_txt_path, 'r', encoding='utf-8') as f:
    case_text = f.read().strip()


# -----------------------------
# If user wants rewritten, call rewrite.py
# -----------------------------
if use_translated_only:
    try:
        from rewrite import translate_case_from_txt
        translate_available = True
    except ImportError:
        translate_available = False

    if translate_available:
        case_text = translate_case_from_txt(case_txt_path)
        print("=== Using translated only case ===\n")
    else:
        print("Translate function not found. Using original case.\n")
elif use_rewritten:
    try:
        from rewrite import rewrite_case_from_txt
        rewrite_available = True
    except ImportError:
        rewrite_available = False

    if rewrite_available:
        case_text = rewrite_case_from_txt(case_txt_path, model=MODEL_NAME)
        print("=== Using rewritten case ===\n")
    else:
        print("Rewrite function not found. Using original case.\n")


# -----------------------------
# Format Simple Request prompt
# -----------------------------
prompt = get_prompt_for_configuration(case_text, 'simple')

# -----------------------------
# Run prompt
# -----------------------------
response = chatgpt_chat_completion(prompt, model=MODEL_NAME)

# -----------------------------
# Output
# -----------------------------
print("=== Case Text ===")
print(case_text)
print("\n=== Simple Request Output ===")
print(response)
