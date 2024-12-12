#이전 수업 시간에 만들었던 이미지 분류 pkl 파일을 바탕으로 한 이미지 분류 모델을 Streamlit에 올리는 예제 코드
#파일 이름 streamlit_app.py

import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown



