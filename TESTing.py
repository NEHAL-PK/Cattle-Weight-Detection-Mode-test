import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the custom metric function
def exact_accuracy(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, tf.keras.backend.round(y_pred)))

# Load the model with the custom metric
model = load_model('best_model.h5', custom_objects={'exact_accuracy': exact_accuracy})

# Define constants
IMG_SIZE = 256  # as defined earlier
CHANNELS = 3

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image at path: {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return (img / 255.0).astype(np.float32)

# Predict weight for an image
image_path = '2_b4-4_s_95_M.jpg'
processed_img = preprocess_image(image_path)
if processed_img is not None:
    prediction = model.predict(np.expand_dims(processed_img, axis=0))
    predicted_weight = prediction[0][0]

    # Display the image with the predicted weight
    img_to_show = cv2.imread(image_path)
    img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    plt.imshow(img_to_show)
    plt.title(f'Predicted Weight: {predicted_weight:.2f}kg')
    plt.show()
else:
    print("Couldn't process the image.")
