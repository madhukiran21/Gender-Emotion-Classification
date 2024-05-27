from flask import Flask, render_template, request, redirect
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

MODEL_PATHS = {
    'Binary Classification': 'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/gender_classifier.h5',
    'CNN Model': 'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier.h5',
    'Multi Class Model': 'G:/Deep Learning/Project/pythonProject3/Proj  ect/Web Application/saved/emotion_classifier_multi_class.h5',
    'VGG16 Model': 'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_vgg16.h5',
    'Regularization Model': 'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_regularization.h5',
    'Sequential Model':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_seq.h5',
    'RNN Model':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_rnn.h5',
    'Seq with OPT':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_seq_opt.h5',
    'Compare 3':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_compare.h5',
    'CNN+LSTM':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_cnn_lstm.h5',
    'Auto Encoder':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_autoencoder.h5',
    'Denoise Auto':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_denoise.h5',
    'Auto+Resnet':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_auto_resnet.h5',
    'VAE':'G:/Deep Learning/Project/pythonProject3/Project/Web Application/saved/emotion_classifier_vae.h5',
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_binary(image_path):
    model_path = MODEL_PATHS['Binary Classification']
    model = tf.keras.models.load_model(model_path)
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    return prediction

def predict(image_path, model_name):
    model_path = MODEL_PATHS[model_name]
    model = tf.keras.models.load_model(model_path)
    image = Image.open(image_path)
    if model_name == 'CNN Model' or model_name == 'Regularization Model':
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_shape = model.input_shape[1:3]
        image = image.resize(input_shape)
        image = np.array(image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
    elif model_name == 'Multi Class Model':
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = Image.open(image_path)
        image = image.resize((48, 48))
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_shape = model.input_shape[1:3]
        image = image.resize(input_shape)
        image = np.array(image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', secure_filename(filename))
            file.save(file_path)
            selected_model = request.form.get('model')
            if selected_model == 'Binary Classification':
                prediction = predict_binary(file_path)
                result_template = ('result1.html')
            else:
                prediction = predict(file_path, selected_model)
                result_template = 'result2.html'

            os.remove(file_path)
            return render_template(result_template, prediction=prediction)
    return render_template('upload.html', models=MODEL_PATHS.keys())

if __name__ == '__main__':
    app.run(debug=True)
