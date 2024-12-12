import streamlit as st
import pickle
import requests
from fastai.learner import load_learner
import pandas as pd
import numpy as np  # numpy를 import해야 함

# Streamlit 제목
st.title("Model Metadata Viewer")

# GitHub Raw 파일 URL과 모델 유형
GITHUB_RAW_URL = "https://github.com/gilsane/classtim/raw/refs/heads/main/xgb_model.pkl"
MODEL_TYPE = "XGBoost"  # "fastai", "scikit-learn Random Forest", or "XGBoost"
CSV_FILE_URL = "https://github.com/gilsane/classtim/raw/refs/heads/main/(%EC%8B%A4%EC%8A%B5%EC%9A%A9)%20%EC%95%84%ED%8C%8C%ED%8A%B8%EA%B0%80%20%EC%8B%A4%EA%B1%B0%EB%9E%98%20%EA%B0%80%EA%B3%B5%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20(1).csv"

# GitHub에서 파일 다운로드 및 로드
def download_model(url, output_path="model.pkl"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            file.write(response.content)
        return output_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

def load_model(file_path, model_type):
    try:
        if model_type == "fastai":
            return load_learner(file_path)  # Fastai 모델 로드
        else:
            with open(file_path, "rb") as file:
                return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# CSV 파일 읽기
def load_csv_with_encodings(url):
    encodings = ["utf-8", "utf-8-sig", "cp949"]
    for encoding in encodings:
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(url, encoding=encoding)
            st.success(f"CSV file loaded successfully with encoding: {encoding}")
            return df
        except Exception as e:
            continue
    st.error("Failed to load CSV file with supported encodings.")
    return None        



# 모델 다운로드 및 로드
downloaded_file = download_model(GITHUB_RAW_URL)
if downloaded_file:
    model = load_model(downloaded_file, MODEL_TYPE)
else:
    model = None

if model is not None:
    st.success("Model loaded successfully!")

# CSV 파일 로드 및 출력
df = load_csv_with_encodings(CSV_FILE_URL)
if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # 사용자 입력 레이아웃 생성
    st.write("### User Input Form")
    col1, col2 = st.columns(2)

    if isinstance(model, dict):  # 모델이 딕셔너리인지 확인
        with col1:
            st.write("**Categorical Features**")
            cat_inputs = {}
            if "cat_names" in model and model["cat_names"]:
                for cat in model["cat_names"]:
                    if cat in df.columns:
                        cat_inputs[cat] = st.selectbox(f"{cat}", options=df[cat].unique())

        with col2:
            st.write("**Continuous Features**")
            cont_inputs = {}
            if "cont_names" in model and model["cont_names"]:
                for cont in model["cont_names"]:
                    if cont in df.columns:
                        cont_inputs[cont] = st.text_input(f"{cont}", value="", placeholder="Enter a number")

        st.write("### User Inputs Summary")
        st.write({"Categorical Inputs": cat_inputs, "Continuous Inputs": cont_inputs})
    else:
        st.error("The loaded model is not in the expected dictionary format.")



    # 예측 버튼 및 결과 출력
    if st.button("Predict"):
        try:
            # 입력 데이터 준비
            input_data = []
            for cat in model["cat_names"]:
                if cat in cat_inputs:
                    input_data.append(cat_inputs[cat])
            for cont in model["cont_names"]:
                if cont in cont_inputs:
                    input_data.append(float(cont_inputs[cont]))  # 숫자로 변환

            # 입력 데이터를 numpy 배열로 변환
            input_array = np.array([input_data])

            # 예측 수행
            prediction = model["model"].predict(input_array)[0]

            # 결과 출력
            y_name = model.get("y_names", ["Prediction"])[0]
            st.success(f"{y_name}: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")



