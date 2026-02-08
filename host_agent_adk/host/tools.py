from pydantic import BaseModel
from typing import Dict, List

# class EvidenceBundle(BaseModel):
#     product_id: str
#     predicted_failure: str
#     confidence_score: str
#     sop_matches: List[Dict[str, str]]
#     recommended_action: str

# class ExplanationResult(BaseModel):
#     summary: str
#     evidence_reasoning: str
#     sop_alignment: str

def generate_explanation_tool(evidence: dict) -> dict:
    """
    Generates a natural-language explanation for a predicted failure.
    Returns a structured dict with four sections:
      1. summary
      2. evidence_reasoning
      3. sop_alignment
      4. confidence_and_limitations
    """
    # 1. Summary
    summary = f"Machine {evidence.get('product_id', 'unknown machine')} shows a high likelihood of {evidence.get('predicted_failure', 'unknown failure')}."

    # 2. Evidence-based reasoning
    evidence_reasoning = (
        f"The predictive model reports a confidence score of {evidence.get('confidence_score', 'N/A')}. "
    )

    # 3. SOP alignment dynamically
    sop_matches = evidence.get("sop_matches", [])
    if sop_matches:
        sop_texts = []
        for sop in sop_matches:
            section = sop.get("section", "N/A")
            description = sop.get("description", "")
            sop_texts.append(f"Section {section}: {description}")
        sop_alignment = " ".join(sop_texts)
    else:
        sop_alignment = "No SOP alignment information available."

    # Return a structured, dynamic explanation
    return {
        "summary": summary,
        "evidence_reasoning": evidence_reasoning,
        "sop_alignment": sop_alignment
    }