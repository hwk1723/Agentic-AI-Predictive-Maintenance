from typing import Dict, Any

# Simple in-memory evidence store keyed by product_id
EVIDENCE_STORE: Dict[str, Dict[str, Any]] = {}

def evidence_collection_tool(product_id: str, source: str, payload: dict) -> Dict[str, Dict[str, Any]]:
    """
    Incrementally collects and stores evidence from multiple agents.

    Args:
        product_id: Unique equipment or machine identifier
        source: Agent providing the evidence (e.g. "maintenance_documentation_agent", "db_agent", "predictive_agent")
        payload: Data from the sub-agents, structured as:
            - raw_sensor_data: List of raw sensor readings (from DB agent)
            - predicted_failure_type: Predicted failure type (from Predictive Maintenance agent)
            - confidence_score: Confidence score of the prediction (from Predictive Maintenance agent)
            - document_matches: List of relevant SOP sections (from RAG agent)

    Returns:
        The updated evidence record for the product_id
    """

    if product_id not in EVIDENCE_STORE:
        EVIDENCE_STORE[product_id] = {
            "product_id": product_id,
            "raw_sensor_data": [],
            "predicted_failure_type": [],
            "confidence_score": [],
            "document_matches": [],
        }

    record = EVIDENCE_STORE[product_id]

    # Normalize known fields from predictive maintenance agent
    normalized_source = source.lower().replace(" ", "_")
    if normalized_source in {"predictive_maintenance_agent", "prediction_agent"}:
        record["predicted_failure_type"].extend(payload["predicted_failure_type"])
        record["confidence_score"].extend(payload["confidence_score"])

    # Normalize SOP matches from RAG agent
    if normalized_source in {"maintenance_documentation_agent", "documentation_agent"}:
        record["document_matches"].extend(payload["document_matches"])

    # Store ONLY raw sensor data from database agent
    if normalized_source in {"db_agent", "database_agent"}:
        record["raw_sensor_data"].extend(payload["raw_sensor_data"])

    return record

def generate_explanation_tool(evidence: Dict[str, Dict[str, Any]]) -> dict:
    """
    Generates a natural-language explanation once sufficient evidence is collected.

    Returns a structured dict with:
      1. summary
      2. document_instructions
    """

    if not evidence:
        return {
            "summary": "No evidence available.",
            "document_instructions": "",
        }

    # 1. Summary
    summary = (
        f"Machine {evidence.get('product_id', 'an unknown ID')} has been analyzed. "
        f"Raw sensor data: {(evidence.get('raw_sensor_data', []))}. "
        f"It is predicted to experience {evidence.get('predicted_failure_type', 'an unknown failure mode')} "
        f"with a confidence score of {evidence.get('confidence_score', 'N/A')}."
    )

    # 3. SOP alignment
    document_matches = evidence.get("document_matches", [])
    if document_matches:
        document_instructions = " ".join(
            f"Section {s.get('section', 'N/A')}: {s.get('description', '')}"
            for s in document_matches
        )
    else:
        document_instructions = "No relevant SOP sections were identified."

    return {
        "summary": summary,
        "document_instructions": document_instructions,
    }