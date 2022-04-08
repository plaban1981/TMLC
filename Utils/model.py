from tensorflow.keras.models import load_model
import numpy as np
model_path = r"Model/keras_final_model.h5"
model = load_model(model_path)
#
def predict(attribues):
    prediction = model.predict(attribues)
    pred = np.argmax(prediction, axis=1)
    return pred