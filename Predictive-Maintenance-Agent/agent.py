import sqlite3
from google.adk.agents import LlmAgent

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def machine_condition_classifier_tool(air_temperature: float, process_temperature: float, rotational_speed: float, torque: float, tool_wear: float, type_: str)-> str:
    """
    machine_condition_classifier_tool

    Predicts the operational condition of an industrial machine based on its sensor data or telemetry readings.

    Use this tool to perform diagnostic classification. The tool leverages a pretrained machine learning model that has been trained on historical machine performance and failure data.

    Inputs:
        Machine Parameters:
            - `air_temperature` (float): The air temperature around the machine in Kelvin.
            - `process_temperature` (float): The process temperature of the machine in Kelvin.
            - `rotational_speed` (float): The rotational speed of the machine in revolutions per minute (rpm).
            - `torque` (float): The torque applied by the machine in Newton-meters (Nm).
            - `tool_wear` (float): The wear level of the machine's tool in minutes.
            - `type_` (str): The type of machine, e.g. 'H', 'L', 'M'. 

    Output:
        str:
            - `predicted_condition` (str): The machines operational state, e.g. "No Failure", "Degraded", "Failure Imminent".

    Example Response:
        "predicted_condition": "No Failure"

    Usage Notes:
        - Do not modify or reinterpret the prediction — rely on the model output.
        - Do not recommend actions at this point
    """

    # ---------- 1) Load checkpoint ----------
    checkpoint_path = "model_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    # ---------- 2) Rebuild the model ----------
    class ClassificationModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(ClassificationModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_dim = checkpoint["input_dim"]
    n_classes = checkpoint["n_classes"]
    model = ClassificationModel(input_dim, n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---------- 3) Rebuild scaler ----------
    scaler_state = checkpoint["scaler_state"]
    mean = np.array(scaler_state["mean"])
    scale = np.array(scaler_state["scale"])
    var = scaler_state.get("var", None)

    # Define a simple scaling function using saved stats
    def scale_input(X_raw):
        return (X_raw - mean) / scale

    # ---------- 4) Rebuild label encoder ----------
    failure_classes = checkpoint["label_encoder_classes"]
    def decode_label(label_id):
        return failure_classes[label_id]

    # ---------- 5) Prepare new sample ----------
    feature_columns = checkpoint["feature_columns"]

    type_H, type_M, type_L = 0, 0, 0

    if type_ == 'H':
        type_H = 1
    elif type_ == 'L':
        type_L = 1
    else:
        type_M = 1

    # Example input sample (replace with your own data)
    new_sample = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": type_H,
        "Type_L": type_L,
        "Type_M": type_M
    }])[feature_columns]  # enforce column order

    # ---------- 6) Scale and convert to tensor ----------
    X_input = torch.tensor(scale_input(new_sample.values), dtype=torch.float32)

    # ---------- 7) Predict ----------
    with torch.no_grad():
        outputs = model(X_input)
        _, predicted = torch.max(outputs, 1)
        predicted_label_id = predicted.item()

    # ---------- 8) Decode label ----------
    predicted_failure_type = decode_label(predicted_label_id)
    # print(f"Predicted Failure Type (numeric): {predicted_label_id}")
    # print(f"Predicted Failure Type (string): {predicted_failure_type}")
    return predicted_failure_type

def create_agent() -> LlmAgent:
    """Constructs the ADK agent for maintenance."""
    return LlmAgent(
        model="gemini-2.5-pro",
        name="Maintenance_Agent",
        instruction="""
            **Role:** You are a maintenance assistant. 
            Your sole responsibility is to predict machine failure types based on operational conditions.

            **Core Directives:**

            *   **Predict Machine Condition:** 
                    Use the `machine_condition_classifier_tool` to predict the failure type based on the operational condition of a machine.
                    The tool requires a `type_` and `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, and `tool_wear`.

            *   **Polite and Concise:** 
                    Always be polite and to the point in your responses.

            *   **Stick to Your Role:** Do not engage in any conversation outside of maintenance. 
                    If asked other questions, politely state that you can only help with maintenance.
        """,
        tools=[machine_condition_classifier_tool],
    )
