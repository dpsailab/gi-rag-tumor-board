"""
Utility functions for interacting with OpenAI Chat Completions
and Assistant API endpoints.

This module is used to execute prompt-based experiments for
the gastrointestinal oncology tumor board study.

IMPORTANT:
- No patient data is stored in this file.
- API keys and Assistant IDs must be provided by the user.
- This code is intended for research purposes only.
"""

import time
from openai import OpenAI

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

# Insert your OpenAI project API key here or load it from environment variables
API_KEY_PROJECT = "YOUR PROJECT API KEY"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY_PROJECT)


# =============================================================================
# CHAT COMPLETION (STANDARD CHAT API)
# =============================================================================

def chatgpt_chat_completion(prompt_text: str, model: str) -> str:
    """
    Send a single-turn prompt to an OpenAI chat completion model.

    This function is used for:
    - Simple request (no retrieval) configurations
    - Custom RAG configurations where retrieved context
      is injected directly into the prompt

    Parameters
    ----------
    prompt_text : str
        Fully formatted prompt string sent to the model.
    model : str
        Model identifier (e.g. "gpt-4o-mini" or "gpt-4o").

    Returns
    -------
    str
        Model-generated response text.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        temperature=0.2,
        top_p=1.0,
    )

    # Extract and return the assistant's response text
    return response.choices[0].message.content.strip()


# =============================================================================
# CHATGPT ASSISTANT API (WITH UPLOADED GUIDELINES)
# =============================================================================

def chatgpt_assistant(prompt_text: str) -> str:
    """
    Send a prompt to a pre-configured ChatGPT Assistant.

    This function is used for the "Assistant" configurations
    where full guideline PDFs are uploaded to the Assistant
    environment and retrieval is handled internally by OpenAI.

    Parameters
    ----------
    prompt_text : str
        Fully formatted prompt string sent to the Assistant.

    Returns
    -------
    str
        Model-generated response text.
    """

    # Insert your Assistant ID here
    ASSISTANT_ID = "YOUR ASSISTANT ID"

    # Create a new conversation thread with the user prompt
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ]
    )

    # Start the Assistant run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )

    print(f"Assistant run created: {run.id}")

    # Poll run status until completion
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run.status == "completed":
            break

        if run.status == "failed":
            raise RuntimeError(
                f"Assistant run failed: {run}"
            )

        print(f"Run status: {run.status}")
        time.sleep(1)

    # Retrieve messages from the completed thread
    message_response = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    # The latest assistant message is the first item
    latest_message = message_response.data[0]

    return latest_message.content[0].text.value.strip()
