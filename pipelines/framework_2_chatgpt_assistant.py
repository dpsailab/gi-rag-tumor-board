"""
Run an ASSISTANT prompt on a patient case (original or rewritten).

User selects whether to use the rewritten case. Retrieval of guidelines
is handled internally by the pre-configured ChatGPT Assistant.

IMPORTANT:
- Model selection is handled on the Assistant side (not here).
- Any guideline PDFs or documents must be uploaded/available in the Assistant environment, if needed.
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
from chatgpt import chatgpt_assistant

# -----------------------------
# User instructions
# -----------------------------
print("NOTE: Model selection is handled internally by the ChatGPT Assistant.")
print("Ensure any guideline PDFs are uploaded to the Assistant if you want them referenced.\n")

# -----------------------------
# User configuration
# -----------------------------
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
    # Temporarily add current folder to sys.path to import rewrite.py
    sys.path.append(current_dir)
    try:
        from rewrite import translate_case_from_txt
        translate_available = True
        case_text = translate_case_from_txt(case_txt_path)
        print("=== Using translated only case ===\n")
    except ImportError:
        translate_available = False
        print("Translate function not found. Using original case.\n")
    finally:
        # Optional: remove current folder from sys.path after import
        if current_dir in sys.path:
            sys.path.remove(current_dir)

elif use_rewritten:
    # Temporarily add current folder to sys.path to import rewrite.py
    sys.path.append(current_dir)
    try:
        from rewrite import rewrite_case_from_txt
        case_text = rewrite_case_from_txt(case_txt_path)
        print("=== Using rewritten case ===\n")
    except ImportError:
        print("Rewrite function not found. Using original case.\n")
    finally:
        # Optional: remove current folder from sys.path after import
        if current_dir in sys.path:
            sys.path.remove(current_dir)

# -----------------------------
# Format Assistant prompt
# -----------------------------
prompt = get_prompt_for_configuration(case_text, 'assistant')

# -----------------------------
# Run prompt using ChatGPT Assistant
# -----------------------------
response = chatgpt_assistant(prompt)

# -----------------------------
# Output
# -----------------------------
print("=== Case Text ===")
print(case_text)
print("\n=== Assistant Prompt Output ===")
print(response)
