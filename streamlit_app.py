import streamlit as st
import pickle
import requests
from fastai.learner import load_learner

# Streamlit 제목
st.title("Model Metadata Viewer")

# GitHub Raw 파일 URL과 모델 유형
GITHUB_RAW_URL = "https://github.com/gilsane/classtim/raw/refs/heads/main/regression_model_fastai.pkl"
MODEL_TYPE = "fastai"  # "fastai", "scikit-learn Random Forest", or "XGBoost"

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

# 모델 다운로드 및 로드
downloaded_file = download_model(GITHUB_RAW_URL)
if downloaded_file:
    model = load_model(downloaded_file, MODEL_TYPE)
else:
    model = None

if model is not None:
    st.success("Model loaded successfully!")

    if MODEL_TYPE == "fastai":
        meta_data = {
            "model": model.model if hasattr(model, 'model') else model,  # 모델의 핵심 내용
            "cat_names": model.dls.cat_names if hasattr(model.dls, 'cat_names') else [],
            "cont_names": model.dls.cont_names if hasattr(model.dls, 'cont_names') else [],
            "y_names": model.dls.y_names if hasattr(model.dls, 'y_names') else [],
            "procs": model.dls.procs if hasattr(model.dls, 'procs') else []
        }
    elif MODEL_TYPE == "scikit-learn Random Forest":
        meta_data = {
            "model": model.get("model", None),  # 모델의 핵심 내용
            "cat_names": model.get("cat_names", []),  # 모델 내 저장된 cat_names 사용
            "cont_names": model.get("cont_names", []),  # 모델 내 저장된 cont_names 사용
            "y_names": model.get("y_names", []),  # 모델 내 저장된 y_names 사용
            "procs": model.get("procs", [])  # 모델 내 저장된 procs 사용
        }
    elif MODEL_TYPE == "XGBoost":
        meta_data = {
            "model": model.get("model", None),  # 모델의 핵심 내용
            "cat_names": model.get("cat_names", []),  # 모델 내 저장된 cat_names 사용
            "cont_names": model.get("cont_names", []),  # 모델 내 저장된 cont_names 사용
            "y_names": model.get("y_names", []),  # 모델 내 저장된 y_names 사용
            "procs": model.get("procs", [])  # 모델 내 저장된 procs 사용
        }
    else:
        st.error("Unsupported model type.")
        meta_data = None

    # 메타 데이터 출력
    if meta_data:
        st.write("### Metadata")
        st.json(meta_data)
