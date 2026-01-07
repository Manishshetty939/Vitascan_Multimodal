from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
DATA_FOLDER = os.path.join(BASE_DIR, 'Data')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def list_models():
    try:
        return sorted([f for f in os.listdir(MODELS_FOLDER) if os.path.isfile(os.path.join(MODELS_FOLDER, f))])
    except Exception:
        return []


def load_symptoms():
    csv_path = os.path.join(DATA_FOLDER, 'symptoms.csv')
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # return header-agnostic list (first column values)
        return [line.split(',')[0] for line in lines]
    except Exception:
        return []


@app.route('/', methods=['GET'])
def index():
    models = list_models()
    symptoms = load_symptoms()
    return render_template('index.html', models=models, symptoms=symptoms)


@app.route('/predict', methods=['POST'])
def predict():
    # Simple handler: save uploaded file and echo selection back.
    file = request.files.get('image')
    selected_symptoms = request.form.getlist('symptoms')
    chosen_model = request.form.get('model')

    filename = None
    if file and file.filename:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

    # Placeholder: real prediction wiring can be added later
    result = {
        'filename': filename,
        'symptoms': selected_symptoms,
        'model': chosen_model,
        'available_models': list_models()
    }

    return render_template('result.html', result=result)


if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True)
