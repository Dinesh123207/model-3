from flask import Flask, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Load the model and class names
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Path to the image (1.png is assumed to be in the same directory as app.py)
image_path = "7b.png"

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Resizing the image to 224x224 and cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Reshape to match model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure the image exists in the same directory as app.py
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file '1.png' not found in the directory"}), 400

        # Preprocess the image
        data = preprocess_image(image_path)

        # Predict the class
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove extra whitespace
        confidence_score = prediction[0][index]

        # Return the class and confidence score
        return jsonify({
            "class": class_name,
            "confidence_score": float(confidence_score)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
