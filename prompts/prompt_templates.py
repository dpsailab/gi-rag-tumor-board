"""
Prompt templates for gastrointestinal oncology RAG tumor board system.

This module contains the four main prompt templates used in the study:
1. Rewriting prompt (German → English standardization)
2. Simple request (no retrieval baseline)
3. ChatGPT Assistant prompt (with uploaded PDFs)
4. Custom RAG prompt (with retrieved guideline chunks)

No patient data or guideline content is included.
"""

from typing import List, Dict, Optional

# =============================================================================
# PROMPT TYPE 1: REWRITING PROMPT
# =============================================================================

REWRITING_PROMPT = """I would like to use the following clinical case in a Retrieval-Augmented Generation (RAG) workflow, referencing oncological guidelines to determine the next therapeutic step.

Please rewrite the case in english in a concise and standardized format, using clinical terminology and structure typically found in oncology guidelines.

Return only the reformulated case, without additional commentary.

#########

Clinical case: {original_case}"""


# =============================================================================
# PROMPT TYPE 1b: TRANSLATE ONLY PROMPT
# =============================================================================

TRANSLATING_PROMPT = """Translate the following clinical case from German into English.

Requirements:
- Preserve the original structure exactly (paragraphs, headings, bullet points, numbering).
- Do not add, remove, or reorder any information.
- Translate all narrative text into clear medical English.
- Keep medical terminology, abbreviations, drug names, and laboratory values unchanged unless a direct translation is required for understanding.
- Do not improve, summarize, or rephrase beyond translation.
- Maintain a formal clinical tone.

Return only the translated case, without any additional commentary or explanation.

#########

Clinical case in German: {original_case}"""


# =============================================================================
# PROMPT TYPE 2: SIMPLE REQUEST (NO RETRIEVAL)
# =============================================================================

SIMPLE_REQUEST_PROMPT = """You are a multidisciplinary oncological board with surgeons, oncologists, radiologists, radiation oncologists, and pathologists. Decide the next therapeutic step for the patient. Patient details are enclosed within triple single quotation marks (''' ''').

If a 'Fragestellung' is present, answer precisely; otherwise, respond based on expertise. Do not suggest referral or further discussion. Answer in German, 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''{case_text}'''"""


# =============================================================================
# PROMPT TYPE 3: CHATGPT ASSISTANT PROMPT
# =============================================================================

ASSISTANT_PROMPT = """You are a multidisciplinary oncological board (surgeons, oncologists, radiologists, radiation oncologists, pathologists). Determine the next therapeutic step based on patient details and attached guidelines.

Patient details under 'Actual Case' (''' '''). Analyze the case in the context of the guidelines (attached files) and provide a concise German recommendation in 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''{case_text}'''"""


# =============================================================================
# PROMPT TYPE 4: CUSTOM RAG PROMPT
# =============================================================================

CUSTOM_RAG_PROMPT = """You are a multidisciplinary oncological board (surgeons, oncologists, radiologists, radiation oncologists, pathologists). Determine the next therapeutic step for the patient based on:

1. Patient clinical details under 'Actual Case' (''' ''')
2. Relevant guideline excerpts under 'Context' (''' ''')

Formulate a concise German recommendation in 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''{case_text}'''

### Context:

'''{retrieved_context}'''"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_retrieved_context(chunks: List[Dict]) -> str:
    """
    Format retrieved guideline chunks into context string.
    
    Args:
        chunks: List of dictionaries with keys 'chunk_id', 'source', 'text'
        
    Returns:
        Formatted context string for insertion into RAG prompt
        
    Example:
        chunks = [
            {'chunk_id': 153, 'source': 'NCCN Guidelines', 'text': '...'},
            {'chunk_id': 157, 'source': 'NCCN Guidelines', 'text': '...'}
        ]
    """
    context_lines = []
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', 'Unknown')
        source = chunk.get('source', 'Unknown source')
        text = chunk.get('text', '')
        context_lines.append(f"- Chunk {chunk_id} - {source}: {text}")
    
    return '\n'.join(context_lines)


def get_prompt_for_configuration(
    case_text: str,
    config_type: str,
    retrieved_chunks: Optional[List[Dict]] = None
) -> str:
    """
    Get the appropriate prompt based on configuration type.
    
    Args:
        case_text: Patient case description (original or rewritten)
        config_type: One of ['simple', 'assistant', 'rag_full', 'rag_selected']
        retrieved_chunks: List of retrieved guideline chunks (for RAG configs only)
        
    Returns:
        Formatted prompt string ready for model execution
        
    Raises:
        ValueError: If config_type is invalid or retrieved_chunks missing for RAG
    """
    if config_type == 'simple':
        return SIMPLE_REQUEST_PROMPT.format(case_text=case_text)
    
    elif config_type == 'assistant':
        return ASSISTANT_PROMPT.format(case_text=case_text)
    
    elif config_type in ['rag_full', 'rag_selected']:
        if retrieved_chunks is None:
            raise ValueError(f"retrieved_chunks required for {config_type}")
        
        context = format_retrieved_context(retrieved_chunks)
        return CUSTOM_RAG_PROMPT.format(
            case_text=case_text,
            retrieved_context=context
        )
    
    else:
        raise ValueError(
            f"Invalid config_type: {config_type}. "
            f"Must be one of: simple, assistant, rag_full, rag_selected"
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# if __name__ == "__main__":
#     # Example case (simplified)
#     case = """64-year-old male with esophageal adenocarcinoma, cT3 cN1 M0. 
#     Completed neoadjuvant chemoradiotherapy 6 weeks ago. 
#     Restaging shows partial response. What is the next step?"""
#     
#     # Example 1: Simple request
#     print("=== SIMPLE REQUEST ===")
#     prompt = get_prompt_for_configuration(case, 'simple')
#     print(prompt)
#     print()
#     
#     
#     # Example 2: RAG with retrieved chunks
#     print("=== CUSTOM RAG ===")
#     chunks = [
#         {
#             'chunk_id': 153,
#             'source': 'NCCN Guidelines',
#             'text': 'Patients with locally advanced esophageal adenocarcinoma should receive neoadjuvant chemoradiotherapy followed by surgery...'
#         },
#         {
#             'chunk_id': 42,
#             'source': 'German S3 Guideline',
#             'text': 'Surgical resection should be performed 4-8 weeks after completion of neoadjuvant therapy...'
#         }
#     ]
#     prompt = get_prompt_for_configuration(case, 'rag_full', chunks)
#     print(prompt)
