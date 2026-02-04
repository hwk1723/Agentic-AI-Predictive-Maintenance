import sqlite3
from google.adk.agents import LlmAgent

DB_PATH = "predictive_maintenance.db"


def query_machine_status(product_id: str) -> str:
    """
    Get machine parameters by product ID.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            type,
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        FROM maintenance
        WHERE product_id = ?
    """, (product_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return f"No machine found with product ID {product_id}"

    type_, air, process, rpm, torque, wear = row

    return (
        f"Product {product_id} ({type_}):\n"
        f"- Air temperature: {air} K\n"
        f"- Process temperature: {process} K\n"
        f"- Speed: {rpm} rpm\n"
        f"- Torque: {torque} Nm\n"
        f"- Tool wear: {wear} min\n"
    )

def create_agent() -> LlmAgent:
    """Constructs the ADK agent for maintenance."""
    return LlmAgent(
        model="gemini-2.5-flash",
        name="Maintenance_Agent",
        instruction="""
            **Role:** You are a maintenance assistant. 
            Your sole responsibility is to query information from a database and respond to inquiries about machine parameters.

            **Core Directives:**

            *   **Check Machine Status:** 
                    Use the `query_machine_status` tool to determine the status of a specific machine.
                    The tool requires a `product_id`. If the user only provides a single product ID, use that.

            *   **Polite and Concise:** 
                    Always be polite and to the point in your responses.

            *   **Stick to Your Role:** Do not engage in any conversation outside of maintenance. 
                    If asked other questions, politely state that you can only help with maintenance.
        """,
        tools=[query_machine_status],
    )
