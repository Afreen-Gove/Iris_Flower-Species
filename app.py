import streamlit as st
import pickle
import numpy as np

model=pickle.load(open('model.pkl', 'rb'))
data=pickle.load(open('iris_data.pkl', 'rb'))

st.title('FLOWER SPECIES')


sepal_length=st.text_input('SepalLength')
sepal_width=st.text_input('SepalWidth')
petal_length=st.text_input('PetalLength')
petal_width=st.text_input('PetalWidth')
if st.button('Submit'):
    arr=(np.array([sepal_length,sepal_width, petal_length, petal_width]).reshape(1,4))
    y_pred=model.predict(arr)
    st.text(y_pred)

