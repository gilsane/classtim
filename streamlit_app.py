import streamlit as st
import pickle
import gdown


# Streamlit 제목
st.title("Model Metadata Viewer")

# Google Drive ID와 모델 유형
GDRIVE_ID = "1-0DMaWxpfuZwR7u8X65dnL01UK4Paiaj"
MODEL_TYPE = "fastai"  # "fastai", "scikit-learn Random Forest", or "XGBoost"

@st.cache_data
# Google Drive에서 파일 다운로드 및 로드
def download_and_load_model(gdrive_id):
    try:
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        output = "model.pkl"
        gdown.download(url, output, quiet=False)
        with open(output, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# 모델 로드
model = download_and_load_model(GDRIVE_ID)

if model is not None:
    st.success("Model loaded successfully!")

    if MODEL_TYPE == "fastai":
        meta_data = {
            "model": model,
            "cat_names": model.dls.cat_names if hasattr(model, 'dls') else [],
            "cont_names": model.dls.cont_names if hasattr(model, 'dls') else [],
            "y_names": model.dls.y_names if hasattr(model, 'dls') else [],
            "procs": model.dls.procs if hasattr(model, 'dls') else []
        }
    elif MODEL_TYPE == "scikit-learn Random Forest":
        meta_data = {
            "model": model,
            "cat_names": [col for col in CONFIG['독립변수'] if CONFIG['데이터 유형'][col] == "범주형"],
            "cont_names": [col for col in CONFIG['독립변수'] if CONFIG['데이터 유형'][col] == "숫자형"],
            "y_names": CONFIG["종속변수"],
            "procs": []  # scikit-learn에는 전처리 정보 없음
        }
    elif MODEL_TYPE == "XGBoost":
        meta_data = {
            "model": model,
            "cat_names": [col for col in CONFIG['독립변수'] if CONFIG['데이터 유형'][col] == "범주형"],
            "cont_names": [col for col in CONFIG['독립변수'] if CONFIG['데이터 유형'][col] == "숫자형"],
            "y_names": CONFIG["종속변수"],
            "procs": ["Categorify", "Normalize"]
        }
    else:
        st.error("Unsupported model type.")
        meta_data = None

    # 메타 데이터 출력
    if meta_data:
        st.write("### Metadata")
        st.json(meta_data)
