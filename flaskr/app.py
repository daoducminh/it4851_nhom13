import os
import re

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from tensorflow.keras.models import load_model

from utils.model_prediction import convert_image, predict

load_dotenv(dotenv_path='.env')

SECRET_KEY = os.getenv('SECRET_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
MONGODB_URI = os.getenv('MONGODB_URI')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
NORMAL_MODEL = load_model('saved_models/trained_cnn1')
client = MongoClient(MONGODB_URI)
collection = client['nlp']['monkeys']


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search-text', methods=['POST'])
def search_text():
    print(MONGODB_URI)
    data = request.get_json()
    text = data['text']
    regx = re.compile(f'.*{text}.*', re.IGNORECASE)
    query = {
        '$or': [
            {'latin_name': {'$regex': regx}},
            {'common_name': {'$regex': regx}}
        ]
    }
    result = list(collection.find(query, {'_id': False}))
    return jsonify(result)


@app.route('/search-image', methods=['POST'])
def search_image():
    if 'file' not in request.files:
        return jsonify({'msg': 'No file input'}), 400
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'msg': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        print(file.filename)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        jpeg_image_path = convert_image(filepath)
        result = predict(model=NORMAL_MODEL, image_path=jpeg_image_path)
        return jsonify({'result': result}), 200
    return jsonify({'msg': 'File not allowed'}), 400


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


if __name__ == '__main__':
    app.run()
