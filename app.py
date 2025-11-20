import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('plant_disease_detection-main/model.h5')
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {
    0: 'Early Blight',
    1: 'Healthy',
    2: 'Late Blight',
    3: 'Pepper Bell Bacterial Spot',
    4: 'Powdery',
    5: 'Rust',
    6: 'Tomato Septoria Leaf Spot'
}

# ✅ Remedies for each disease
remedies = {
    "Early Blight":
        "• Remove infected leaves\n"
        "• Use copper-based fungicide\n"
        "• Avoid overhead watering\n"
        "• Improve air circulation",

    "Healthy":
        "• Your plant is healthy!\n"
        "• Continue normal watering & sunlight\n"
        "• Monitor weekly for any changes",

    "Late Blight":
        "• Remove heavily infected leaves\n"
        "• Apply fungicide containing Chlorothalonil\n"
        "• Avoid wetting leaves\n"
        "• Keep plants well spaced",

    "Pepper Bell Bacterial Spot":
        "• Remove infected leaves\n"
        "• Use copper fungicide weekly\n"
        "• Avoid touching plants when wet",

    "Powdery":
        "• Spray mix: 1 tbsp baking soda + 1L water\n"
        "• Increase sunlight & airflow\n"
        "• Remove heavily infected leaves",

    "Rust":
        "• Remove rusted leaves\n"
        "• Use sulfur-based fungicide\n"
        "• Water soil, not leaves",

    "Tomato Septoria Leaf Spot":
        "• Prune lower leaves\n"
        "• Use fungicide (Chlorothalonil or Copper)\n"
        "• Keep foliage dry\n"
        "• Mulch soil to prevent splash infection"
}

def getResult(image_path):
    print("Loading image...")
    img = load_img(image_path, target_size=(225, 225))

    print("Converting to array...")
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    print("Running prediction...")
    predictions = model.predict(x)
    print("Prediction done.")

    return predictions[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filename = secure_filename(f.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        f.save(file_path)
        print("Saved at:", file_path)

        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        remedy = remedies[predicted_label]

        # You can return a combined string OR render an HTML template.
        return f"Disease: {predicted_label}\n\nRemedy:\n{remedy}"

    return None


if __name__ == '__main__':
    app.run(debug=True)
