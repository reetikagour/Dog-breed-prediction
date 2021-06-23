from flask import Flask, request, jsonify , render_template
from keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import pickle

app = Flask(__name__)
model = load_model('model.h5')

with open('breed_indices_dict.pickle', 'rb') as handle:
    breed_indices_mapping = pickle.load(handle)
breed_indices_mapping = {y:x for x,y in breed_indices_mapping.items()}

def process_image(image):
    image = image.resize((224,224))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
    
def predict():
    
    img = request.files['image']
    # img = base64.b64decode(img.read())
    # print(type(img))
    # img.decode('base64')
    image = Image.open(img)
    
    # image = Image.open(BytesIO(base64.b64decode(str(img))))
    processed_image = process_image(image)
    result = model.predict(processed_image)
    print(result)
    predicted_score = max(result[0])
    print(predicted_score)
    index = np.argmax(result, axis=-1)
    print(index)
    breed = breed_indices_mapping[index[0]]
    response = {
        'prediction':{
            'Detected Breed': breed,
            'predicted score': str(predicted_score)
        }
    }    
    return jsonify(response)

@app.route("/")
def index():
    return render_template("index.html");   

if __name__ == "__main__":
    app.run(debug=True)