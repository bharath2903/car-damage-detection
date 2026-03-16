import streamlit as st
from model_helper import predict
st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload Your Image ",type=['jpg','png'])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path,'wb') as f:
        f.write(uploaded_file.getbuffer())

        st.image(uploaded_file,caption="uploaded Picture",use_container_width=True,width=100)

        prediction = predict(image_path)
        st.info(f"Predicted class is {prediction}")