import flask
import numpy as np
import tensorflow.lite as tflite
import cv2
import json
from flask import Flask, request, jsonify

# Load class labels
with open("labels.json", "r") as f:
    labels = json.load(f)
    label_dict = {i: label["name"] for i, label in enumerate(labels)}

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # Expected input image shape

app = Flask(__name__)

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
    
    return jsonify({"label": predicted_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
