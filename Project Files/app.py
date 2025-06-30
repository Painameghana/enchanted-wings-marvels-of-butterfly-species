from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model ONCE at startup!
MODEL_PATH = 'models/vgg16_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = load_model(MODEL_PATH)

butterfly_names = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88', 4: 'APOLLO',
    5: 'ATALA', 6: 'BANDED ORANGE HELICONIAN', 7: 'BANDED PEACOCK', 8: 'BECKERS WHITE',
    9: 'BLACK HAIRSTREAK', 10: 'BLUE MORPHO', 11: 'BLUE SPOTTED CROW', 12: 'BROWN SIPROETA',
    13: 'CABBAGE WHITE', 14: 'CAIRNS BIRDWING', 15: 'CHEQUERED SKIPPER', 16: 'CHESTNUT',
    17: 'CLEOPATRA', 18: 'CLODIUS PARNASSIAN', 19: 'CLOUDED SULPHUR', 20: 'COMMON BANDED AWL',
    21: 'COMMON WOOD-NYMPH', 22: 'COPPER TAIL', 23: 'CRECENT', 24: 'CRIMSON PATCH',
    25: 'DANAID EGGFLY', 26: 'EASTERN COMA', 27: 'EASTERN DAPPLE WHITE', 28: 'EASTERN PINE ELFIN',
    29: 'ELBOWED PIERROT', 30: 'GOLD BANDED', 31: 'GREAT EGGFLY', 32: 'GREAT JAY',
    33: 'GREEN CELLED CATTLEHEART', 34: 'GREY HAIRSTREAK', 35: 'INDRA SWALLOW', 36: 'IPHICLUS SISTER',
    37: 'JULIA', 38: 'LARGE MARBLE', 39: 'MALACHITE', 40: 'MANGROVE SKIPPER', 41: 'MESTRA',
    42: 'METALMARK', 43: 'MILBERTS TORTOISESHELL', 44: 'MONARCH', 45: 'MOURNING CLOAK',
    46: 'ORANGE OAKLEAF', 47: 'ORANGE TIP', 48: 'ORCHARD SWALLOW', 49: 'PAINTED LADY',
    50: 'PAPER KITE', 51: 'PEACOCK', 52: 'PINE WHITE', 53: 'PIPEVINE SWALLOW', 54: 'POPINJAY',
    55: 'PURPLE HAIRSTREAK', 56: 'PURPLISH COPPER', 57: 'QUESTION MARK', 58: 'RED ADMIRAL',
    59: 'RED CRACKER', 60: 'RED POSTMAN', 61: 'RED SPOTTED PURPLE', 62: 'SCARCE SWALLOW',
    63: 'SILVER SPOT SKIPPER', 64: 'SLEEPY ORANGE', 65: 'SOOTYWING', 66: 'SOUTHERN DOGFACE',
    67: 'STRIATED QUEEN', 68: 'TROPICAL LEAFWING', 69: 'TWO BARRED FLASHER', 70: 'ULYSES',
    71: 'VICEROY', 72: 'WOOD SATYR', 73: 'YELLOW SWALLOW TAIL', 74: 'ZEBRA LONG WING'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    predictions = []
    for key in request.files:
        file = request.files[key]
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(224, 224))
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            pred = model.predict(arr)
            idx = np.argmax(pred, axis=1)[0]
            name = butterfly_names.get(idx, "Unknown")
            predictions.append({
                'prediction': name,
                'user_image': url_for('static', filename='images/' + file.filename)
            })
    return render_template('output.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
