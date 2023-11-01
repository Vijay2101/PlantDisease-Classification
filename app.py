import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import cv2

from tensorflow.keras.models import load_model
model = tf.keras.models.load_model('pdclassifier.h5')

label = {'Apple__black_rot': 0,
 'Apple__healthy': 1,
 'Apple__rust': 2,
 'Apple__scab': 3,
 'Cassava__bacterial_blight': 4,
 'Cassava__brown_streak_disease': 5,
 'Cassava__green_mottle': 6,
 'Cassava__healthy': 7,
 'Cassava__mosaic_disease': 8,
 'Cherry__healthy': 9,
 'Cherry__powdery_mildew': 10,
 'Chili__healthy': 11,
 'Chili__leaf curl': 12,
 'Chili__leaf spot': 13,
 'Chili__whitefly': 14,
 'Chili__yellowish': 15,
 'Coffee__cercospora_leaf_spot': 16,
 'Coffee__healthy': 17,
 'Coffee__red_spider_mite': 18,
 'Coffee__rust': 19,
 'Corn__common_rust': 20,
 'Corn__gray_leaf_spot': 21,
 'Corn__healthy': 22,
 'Corn__northern_leaf_blight': 23,
 'Cucumber__diseased': 24,
 'Cucumber__healthy': 25,
 'Gauva__diseased': 26,
 'Gauva__healthy': 27,
 'Grape__black_measles': 28,
 'Grape__black_rot': 29,
 'Grape__healthy': 30,
 'Grape__leaf_blight_(isariopsis_leaf_spot)': 31,
 'Jamun__diseased': 32,
 'Jamun__healthy': 33,
 'Lemon__diseased': 34,
 'Lemon__healthy': 35,
 'Mango__diseased': 36,
 'Mango__healthy': 37,
 'Peach__bacterial_spot': 38,
 'Peach__healthy': 39,
 'Pepper_bell__bacterial_spot': 40,
 'Pepper_bell__healthy': 41,
 'Pomegranate__diseased': 42,
 'Pomegranate__healthy': 43,
 'Potato__early_blight': 44,
 'Potato__healthy': 45,
 'Potato__late_blight': 46,
 'Rice__brown_spot': 47,
 'Rice__healthy': 48,
 'Rice__hispa': 49,
 'Rice__leaf_blast': 50,
 'Rice__neck_blast': 51,
 'Soybean__bacterial_blight': 52,
 'Soybean__caterpillar': 53,
 'Soybean__diabrotica_speciosa': 54,
 'Soybean__downy_mildew': 55,
 'Soybean__healthy': 56,
 'Soybean__mosaic_virus': 57,
 'Soybean__powdery_mildew': 58,
 'Soybean__rust': 59,
 'Soybean__southern_blight': 60,
 'Strawberry___leaf_scorch': 61,
 'Strawberry__healthy': 62,
 'Sugarcane__bacterial_blight': 63,
 'Sugarcane__healthy': 64,
 'Sugarcane__red_rot': 65,
 'Sugarcane__red_stripe': 66,
 'Sugarcane__rust': 67,
 'Tea__algal_leaf': 68,
 'Tea__anthracnose': 69,
 'Tea__bird_eye_spot': 70,
 'Tea__brown_blight': 71,
 'Tea__healthy': 72,
 'Tea__red_leaf_spot': 73,
 'Tomato__bacterial_spot': 74,
 'Tomato__early_blight': 75,
 'Tomato__healthy': 76,
 'Tomato__late_blight': 77,
 'Tomato__leaf_mold': 78,
 'Tomato__mosaic_virus': 79,
 'Tomato__septoria_leaf_spot': 80,
 'Tomato__spider_mites_(two_spotted_spider_mite)': 81,
 'Tomato__target_spot': 82,
 'Tomato__yellow_leaf_curl_virus': 83,
 'Wheat__brown_rust': 84,
 'Wheat__healthy': 85,
 'Wheat__septoria': 86,
 'Wheat__yellow_rust': 87}
list = list(label.keys())

def prediction(model, img):
  resize = tf.image.resize(img, (128,128))
  yhat = model.predict(np.expand_dims(resize/255, 0))
  prediction= list[np.argmax(yhat)]
  return prediction

st.title("Plant Disease Classification")

st.write("This application can predict 88 various types of plant disease of different plants from the image of leaf of the plant.")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)

    # Define the desired image width and height
    desired_width = 400  # Adjust to your preferred width
    desired_height = 400  # Adjust to your preferred height

    # Resize the image to the desired dimensions
    image = img.resize((desired_width, desired_height))

    col1, col2 = st.columns(2)
    with col1:
      # Display the uploaded image with the specified width
      st.image(image, caption="Uploaded Image")
    with col2:
      # Prediction
      predict = prediction(model, img)
      # Display the image name below
      st.write(f"Image Name: {uploaded_image.name}")
      st.text("")
      st.write(f"Prediction : {predict}")