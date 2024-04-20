from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model('models/model.h5')

FLOWER_CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


@app.route("/predict", methods=['POST'])
def predict_flower():
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0  
    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    print(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class = FLOWER_CLASSES[predicted_class_index]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8000)