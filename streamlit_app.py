import streamlit as st
import pandas as pd
import numpy as np

#csv 데이터 파일 경로(Github) 및 모델 파일 경로(Google Drive) 설정

file_path = "https://github.com/gilsane/classtim/raw/refs/heads/main/(%EC%8B%A4%EC%8A%B5%EC%9A%A9)%20%EC%95%84%ED%8C%8C%ED%8A%B8%EA%B0%80%20%EC%8B%A4%EA%B1%B0%EB%9E%98%20%EA%B0%80%EA%B3%B5%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20(1).csv"
model_path = "1-9ILaJ0gX7a4aHNUUOWhY-vmp49-7KwX"
url = f"https://drive.google.com/uc?id={model_path}"

# 파일 읽기
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except Exception:
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding='cp949')
        except Exception:
            with output:
                st.write("모델을 읽어올 수 없습니다. 모델 파일 정보를 다시 한번 확인해주세요")

st.dataframe(df.head()) 

#@st.cache_data
output = "model.pkl"
gdown.download(url, output, quiet=False)

try:
    with open(output, 'rb') as f:
        model_metadata = pickle.load(f)
    model = model_metadata
except Exception as e:
    raise ValueError(f"Scikit-learn 모델 로드 실패: {e}")
else:
    raise ValueError(f"알 수 없는 모델 타입: {model_type}")


