import os
import json
import numpy as np
import tensorflow.lite as tflite
import cv2
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List
from google import genai

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
if not genai_api_key:
    raise ValueError("Missing Gemini API key. Set GOOGLE_GENAI_API_KEY in .env.")

# Gemini client
client = genai.Client(api_key=genai_api_key)

# Pydantic schema for validation
class ResponseSchema(BaseModel):
    Description: str
    Causes: List[str]
    RiskFactors: List[str]
    Prognosis: str
    Treatments: List[str]

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)
    label_dict = {i: label["name"] for i, label in enumerate(labels)}

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Flask app
app = Flask(__name__)

# Function to fetch additional info using Gemini
def fetch_info_from_genai(cancer_type):
    prompts = {
        "Description": f"Provide a concise description (5-10 sentences) of {cancer_type} without labels or disclaimers.",
        "Causes": f"List the causes of {cancer_type} separated by commas, no extra text.",
        "Risk Factors": f"List the risk factors of {cancer_type} separated by commas, no extra text.",
        "Prognosis": f"Provide the average prognosis (e.g., survivability) of {cancer_type} in a single sentence.",
        "Treatments": f"List the treatments for {cancer_type}, one per line, no extra text."
    }

    info = {}
    try:
        for key, prompt in prompts.items():
            response = client.models.generate_content(
                model="gemini-2.0-pro",  # or gemini-2.0-flash
                contents=[prompt],
                config={"response_mime_type": "text/plain"}
            )
            text = response.text.strip()
            if key == "Treatments":
                info[key] = [line.strip() for line in text.splitlines() if line.strip()]
            elif key in ["Causes", "Risk Factors"]:
                info[key] = [item.strip() for item in text.split(",") if item.strip()]
            else:
                info[key] = text

        structured = ResponseSchema(**info)
        return structured.dict()

    except ValidationError as e:
        print(f"[Validation Error] {e}")
    except Exception as e:
        print(f"[Gemini API Error] {e}")

    # fallback
    return {
        "Description": "Information not available.",
        "Causes": [],
        "RiskFactors": [],
        "Prognosis": "Information not available.",
        "Treatments": []
    }

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data)
    predicted_label = label_dict.get(predicted_index, "Unknown")
    confidence = float(np.max(output_data))

    # Fetch extra info
    extra_info = fetch_info_from_genai(predicted_label)

    return jsonify({
        "label": predicted_label,
        "confidence": confidence,
        "details": extra_info
    })

if __name__ == "__main__":
    app.run(debug=True)
