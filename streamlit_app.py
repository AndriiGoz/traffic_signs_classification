import numpy as np
import streamlit as st
import cv2 as cv
from PIL import Image
from keras.models import load_model


# Label traffic signs
labels_dict = {
    0: '274. Speed limit (20km/h)',
    1: '274. Speed limit (30km/h)',
    2: '274. Speed limit (50km/h)',
    3: '274. Speed limit (60km/h)',
    4: '274. Speed limit (70km/h)',
    5: '274. Speed limit (80km/h)',
    6: '278. End of speed limit (80km/h)',
    7: '274. Speed limit (100km/h)',
    8: '274. Speed limit (120km/h)',
    9: '276. No passing',
    10: '277. No passing vehicle over 3.5 tons',
    11: '301. Right-of-way at intersection',
    12: '306. Priority road',
    13: '205. Yield',
    14: '206. Stop',
    15: '250. No vehicles',
    16: '253. Vehicle > 3.5 tons prohibited',
    17: '267. No entry',
    18: '101. General caution',
    19: '104. Dangerous curve left',
    20: '103. Dangerous curve right',
    21: '105-10. Double curve',
    22: '112. Bumpy road',
    23: '114. Slippery road',
    24: '121. Road narrows on the right',
    25: '123. Road work',
    26: '131. Traffic signals',
    27: '133. Pedestrians',
    28: '136. Children crossing',
    29: '138. Bicycles crossing',
    30: '101-51. Beware of ice/snow',
    31: '142. Wild animals crossing',
    32: '282. End speed + passing limits',
    33: '209-20. Turn right ahead',
    34: '209-10. Turn left ahead',
    35: '209-30. Ahead only',
    36: '214-20. Go straight or right',
    37: '214-10. Go straight or left',
    38: '222-20. Keep right',
    39: '222-10. Keep left',
    40: '215. Roundabout mandatory',
    41: '280. End of no passing',
    42: '281. End no passing vehicle > 3.5 tons'
}


@st.cache
def sign_predict(image):
    model = load_model('./keras_model/')
    image = np.array(image, dtype=np.float32)
    image = image/255
    image = np.reshape(image, (1, 32, 32))
    x = image.astype(np.float32)
    prediction = model.predict(x)
    prediction_max = np.argmax(prediction)
    prediction_label = labels_dict[prediction_max]
    confidence = np.max(prediction)
    return prediction_label, confidence


def main():
    # Set page config and markdowns
    st.set_page_config(page_title='Traffic Signs Classifier', page_icon=':car:')
    st.title('Traffic Signs Classifier')
    st.markdown("""
        This application classifies traffic signs. Upload any photo of a traffic sign 
        and receive its name out of 43 present classes. For getting the correct prediction, 
        try to upload a square picture containing only the sign.
        """)
    with st.expander("See list of classes"):
        st.write(list(labels_dict.values()))
    st.image('./app_images/road_sign.jpg', use_column_width=True)
    image_usr = st.file_uploader('Upload a photo of traffic sign here', type=['jpg', 'jpeg', 'png'])

    if image_usr is not None:
        col1, col2 = st.columns(2)
        col1.markdown('#### Your picture')
        col2.markdown('#### Your picture 32x32 gray')
        image = Image.open(image_usr)
        with col1:
            st.image(image, use_column_width=True)

        image_np = np.array(image.convert('RGB'))
        image_col = cv.cvtColor(image_np, 1)
        image_gray = cv.cvtColor(image_col, cv.COLOR_BGR2GRAY)
        image_32 = cv.resize(image_gray, (32, 32))
        with col2:
            st.image(image_32, use_column_width=True)

        # Make prediction
        prediction_label, confidence = sign_predict(image_32)

        st.write('##### Prediction:', prediction_label)
        st.write('##### Confidence:', str(confidence))
        st.markdown('***')

    # Markdowns
    st.subheader('About this app')
    st.markdown("""
    The app uses an implementation of LeNet-5 Convolutional Neural Network. 
    The model was trained and tested on about 40.000 real photos of 43 types of german traffic signs.
    
    Data was taken from The German Traffic Sign Recognition Benchmark (GTSRB):
    https://benchmark.ini.rub.de/gtsrb_dataset.html
    
    Source code on GitHub: https://github.com/AndriiGoz/traffic_signs_classification
    
    Author: Andrii Gozhulovskyi
    """)


if __name__ == '__main__':
    main()
