#potato leaf disease identification and classification
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Potato Leaf Disease Identification and Classification')

def main() :
    file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, recommendation, confidence = predict_class(image)
        st.write('PREDICTION : {}'.format(result))
        st.write('CONFIDENCE : {}%'.format(confidence))
        st.write('RECOMMENDATION :{}'.format(recommendation))
       
def predict_class(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'potatoes.h5', compile = False)

    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
    recom_names = ['Try to maintain optimum growing conditions, including proper fertilization, irrigation, and management of other pests.Spray protectant fungicides for early blight management', 'Eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. Air drainage to facilitate the drying of foliage each day is important.', 'Maintain even moisture, especially from the time after the flowers bloom.Constantly monitor for any infections']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    recommendation = recom_names[np.argmax(prediction)]
    return final_pred,recommendation,confidence

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p align="center"> <a href="https://www.masboy.com//">mas boy in action</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' :
    main()
