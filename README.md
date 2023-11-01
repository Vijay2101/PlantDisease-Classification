
# Plant Disease Classification with Streamlit

This Streamlit application can predict 88 different types of plant diseases from images of plant leaves. It uses a pre-trained TensorFlow model based on the Xception architecture, and it achieves a cross-validation accuracy of 91.07%.




## Demo

You can also access a live demo of the Plant Disease Classification App by clicking [here](https://plantdisease-classification.streamlit.app/).



## Features

- **Plant Disease Prediction**: Upload an image of a plant leaf, and the application will predict the type of disease affecting the plant.

- **Wide Disease Coverage**: The application is trained to recognize and classify 88 different types of plant diseases, making it a comprehensive tool for plant health assessment.

- **User-Friendly Interface**: Streamlit provides an intuitive and user-friendly web interface, making it easy for users to interact with the application.



## Getting Started

- Clone this repository to your local machine.

- Install the required packages using the following command:

```bash
  pip install -r requirements.txt

```

## Usage


- Run the Streamlit app using the following command:
```bash
streamlit run app.py
```

- The application will be available in your web browser.
- Click the "Upload an image" button and select an image of a plant leaf.
- The app will display the uploaded image and predict the plant disease.
- Access the chatbot application by clicking on the following link:

   (https://plantdisease-classification.streamlit.app/)


## Data and Model

This application utilizes the "Plant Disease Classification Merged Dataset" for training and testing. The dataset is a comprehensive collection of plant disease images from various sources.

- **Dataset Source**: [Link to the dataset source](https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset)
- **Dataset Description**: The "Plant Disease Classification Merged Dataset" contains images of plant leaves affected by a wide range of diseases, which are essential for training the plant disease classification model used in this application.

- **Model Architecture**: The model is based on the Xception architecture, a powerful deep learning model.
- **Training Data**: The model was trained on the PlantVillage dataset, which contains a wide variety of plant disease images.
- **Accuracy**: The model achieved an impressive cross-validation accuracy of 91.07%.


## Tech Stack

The Plant Disease Classification application is built using the following technologies and tools:

- **Streamlit**: A Python library for creating web applications with minimal effort.
- **TensorFlow**: An open-source machine learning framework for building and training machine learning models.
- **Xception**: The deep learning model architecture used for image classification.
- **Python**: The primary programming language for developing the application.
- **PIL (Pillow)**: Python Imaging Library for working with images.
- **OpenCV**: An open-source computer vision library used for image processing.




## Credits

This app was created by Vijay Kumar. The Plant Disease Classification Merged Dataset used in this application was sourced from Kaggle. I would like to express my gratitude to the Kaggle community for providing this valuable dataset. 

- Dataset Source: [Plant Disease Classification Merged Dataset](https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset) 
