import os
import re

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from imutils import paths
from random import shuffle
from tensorflow.keras.models import load_model

from utils.model_prediction import convert_image, predict

load_dotenv(dotenv_path='.env')

SECRET_KEY = os.getenv('SECRET_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
MONGODB_URI = os.getenv('MONGODB_URI')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TRAINED_MODEL_PATH = os.getenv('TRAINED_MODEL_PATH')

NORMAL_MODEL = load_model(TRAINED_MODEL_PATH)
IMAGE_FOLDER_REAL_PATH = 'flaskr/static/images/monkeys/{}'
client = MongoClient(MONGODB_URI)
collection = client['nlp']['monkeys']


app = Flask(__name__, static_url_path='/static')
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
    data = request.get_json()
    print(data)
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
        return jsonify({'error': 'No file input'})
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        jpeg_image_path = convert_image(filepath)
        prediction = predict(model=NORMAL_MODEL, image_path=jpeg_image_path)

        if prediction:
            animal_id = prediction[0]
            query = {'id': animal_id}
            result = dict(list(collection.find(query, {'_id': False}))[0])
            try:
                image_paths = list(
                    paths.list_images(IMAGE_FOLDER_REAL_PATH.format(animal_id)))
                shuffle(image_paths)
                sample_images = []
                for image_path in image_paths:
                    sample_images.append(image_path.replace('flaskr', ''))
                result['sample_images'] = sample_images
            except:
                pass
            return jsonify(result)
        else:
            jsonify({'error': 'No result found'})
    return jsonify({'error': 'File not allowed'})


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


if __name__ == '__main__':
    app.run()
