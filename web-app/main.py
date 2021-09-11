from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model.predict import predict
import numpy as np
import base64
from PIL import Image
import os
from werkzeug.utils import secure_filename


app = Flask("rock_paper_scissors")
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        image = Image.open(file_path)
        image = np.array(image)
        result = predict(image)
        return render_template('main.html', result=result)
    return render_template('main.html')


# @app.route('/', methods=['POST'])
# def main():
#     key_dict = request.get_json()
#     image = key_dict["image"]
#     imgdata = base64.b64decode(image)
#     model = Network()
#     vehicle = predict_vehicle(model, imgdata)
#     response = {
#         "result": vehicle,
#     }
#     response = jsonify(response)
#     return response


# @app.route("/predict", methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(file.filename))
#         print("\n\nFile Path\n\n")
#         file.save(file_path)

#         image = Image.open(file_path)
#         print(image)
#         model = Network()
#         vehicle = predict_vehicle(model, image)
#     return render_template('predict.html', vehicle=vehicle)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
