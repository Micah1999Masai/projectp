# Potato Leaf Disease Identification and Classification
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Potato Leaf Disease Identification and Classification')

def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        
        # Display the uploaded image
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        
        # Make predictions
        result, recommendation, confidence = predict_class(image)
        st.write(f'PREDICTION: {result}')
        st.write(f'CONFIDENCE: {confidence}%')
        st.write(f'RECOMMENDATION: {recommendation}')

def predict_class(image):
    # Load the Keras model
    with st.spinner('Loading Model...'):
        model = keras.models.load_model('potatoes.h5', compile=False)

    # Preprocess the image
    test_image = image.resize((256, 256))
    from tensorflow.keras.utils import img_to_array
    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    # Define class names and recommendations
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
    recom_names = [
        'Maintain optimum conditions; use fungicides for early blight.',
        'Remove infected potatoes; use fungicides for late blight.',
        'Ensure consistent moisture; monitor for infections.',
    ]

    # Predict the class
    prediction = model.predict(test_image)
    confidence = round(100 * np.max(prediction[0]), 2)
    final_pred = class_name[np.argmax(prediction)]
    recommendation = recom_names[np.argmax(prediction)]

    return final_pred, recommendation, confidence

if __name__ == '__main__':
    main()
