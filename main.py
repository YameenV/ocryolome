import streamlit as st
import cv2
import numpy as np
from helper import (modelLoad, 
                    imageImport,
                    areaDetection,
                    textDetection,
                    cropImages,
                    )
from pytesseract import image_to_string
from Address import parser


st.title("Welcome to Smart auto fill form")

uploaded_file = st.file_uploader("Upload your Adharcard", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")

    model = modelLoad('./best.pt')
    
    df = areaDetection(img,model)
    images = cropImages(df,img)
    
    for i in df.index:
        data = df[df["class"] == df["class"][i]]
        cv2.rectangle(img,(int(data["xmin"][i]),int(data["ymin"][i])),(int(data["xmax"][i]),int(data["ymax"][i])),(0,255,0),2)
    st.image(img, channels="BGR")

    data = ''
    for imgs in images:
        txt = image_to_string(imgs)
        st.image(imgs)
        st.text(txt)
        data += ' '+ txt

    _dict = parser(data)
    st.text(_dict)

    


