#이전 수업 시간에 만들었던 이미지 분류 pkl 파일을 바탕으로 한 이미지 분류 모델을 Streamlit에 올리는 예제 코드
#파일 이름 streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))

st.dataframe(df)  # Same as st.write(df)
