import tensorflow as tf

def load_model(model_path):

    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

def predict_image(image_path, model):

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)


    prediction = model.predict(image)


    labels = ['label1', 'label2', 'label3']
    predicted_label_index = tf.argmax(prediction[0]).numpy()
    predicted_label = labels[predicted_label_index]

    return predicted_label
