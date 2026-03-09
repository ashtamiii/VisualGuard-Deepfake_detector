import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 224

model = tf.keras.models.load_model("models/deepfake_spatial_model.keras")

def predict_image(path):

    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0

    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "REAL"
    else:
        label = "FAKE"

    print("Prediction:",label)
    print("Confidence:",float(pred))

predict_image(r"C:\Users\ashta\VisualGuard-Deepfake-Detection\dataset\Celeb-DF Preprocessed\test\real\00007_frame60_face5.jpg")