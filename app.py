#antes de ejecutar el codigo asegurase de entrar al carpeta del proyecto
#instalar las dependencias necesarias 
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

model = load_model("face_model.h5")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "archive(6)/train",
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=32
)
class_names = train_ds.class_names

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = None
    img_path = None

    if "cam_image" in request.form and request.form["cam_image"] != "":
        img_data = request.form["cam_image"]
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("L")  # gris
        img = np.array(img)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], "cam_image.jpg")
        Image.fromarray(img).save(img_path)

    elif "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "No se recibió ninguna imagen"

    img_resized = cv2.resize(img, (48, 48))
    img_input = img_resized.reshape(1, 48, 48, 1)

    pred = np.argmax(model.predict(img_input), 1)[0]
    pred_class = class_names[pred]

    relative_path = img_path.replace("static/", "")

    return render_template(
        'index.html',
        prediction=pred_class,
        img_path=f"/static/{relative_path}"
    )


if __name__ == '__main__':
    app.run(debug=True)
