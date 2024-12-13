from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

MODEL_PATH = 'Model.h5'

try:
    print(" ** Loading Model **")
    model = load_model(MODEL_PATH)
    print(" ** Model Loaded Successfully **")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path)
        img_array = np.array(img)
        open_cv_image = cv2.resize(img_array, (256, 256))
        open_cv_image = open_cv_image / 255.0
        open_cv_image = np.expand_dims(open_cv_image, axis=0)

        preds = model.predict(open_cv_image)
        d = preds.flatten()
        max_prob = d.max()
        labels = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
            'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        for index, prob in enumerate(d):
            if prob == max_prob:
                class_name = labels[index].split('___')
                return class_name
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])

def upload():
    try:
        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'uploads')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, secure_filename(f.filename))
            f.save(file_path)

            if model:
                class_name = model_predict(file_path, model)
                if class_name:
                    result = (
                        f"Predicted Crop: {class_name[0]}  "
                        f"Predicted Disease: {class_name[1].title().replace('_', ' ')}"
                    )
                else:
                    result = "Error during prediction. Check the uploaded file format."
            else:
                result = "Model not loaded. Cannot make predictions."

            return result
    except Exception as e:
        return f"Error: {e}"
    return "Invalid request method."

if __name__ == '__main__':
    app.run()
