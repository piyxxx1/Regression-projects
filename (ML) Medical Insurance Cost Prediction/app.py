import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Inject Tailwind & Bootstrap CSS
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    
    <style>
        .main-container {
            padding: 2rem;
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            border-radius: 1rem;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .title {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #0f172a;
        }
        .btn-predict {
            background-color: #0d6efd;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            border: none;
        }
        .btn-predict:hover {
            background-color: #0b5ed7;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}
@keyframes blink {
  50% { border-color: transparent }
}
.typing-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 2.5rem;
  font-weight: 800;
}
.typing-title {
  white-space: nowrap;
  overflow: hidden;
  border-right: 4px solid #3b82f6;
  width: 0;
  animation:
    typing 4s steps(40, end) forwards,
    blink 1s step-end 4;
  background: linear-gradient(to right, #06b6d4, #3b82f6, #8b5cf6, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>

<div class="typing-wrapper">
  <span style="color: black;">🏥</span>
  <span class="typing-title"> Medical Insurance Cost Predictor</span>
</div>
""", unsafe_allow_html=True)


st.write("Enter the details below to predict your estimated insurance charges.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()

def preprocess_data(data):
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()

    data['sex'] = le_sex.fit_transform(data['sex'])
    data['smoker'] = le_smoker.fit_transform(data['smoker'])
    data['region'] = le_region.fit_transform(data['region'])

    return data, le_sex, le_smoker, le_region

df, le_sex, le_smoker, le_region = preprocess_data(df)

X = df.drop(columns='charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Form layout
with st.form("prediction_form"):
    st.subheader("📝 Input Features")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
        
    with col2:
        sex = st.radio("Sex", ['male', 'female'])
        smoker = st.radio("Smoker", ['yes', 'no'])
        region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

    submitted = st.form_submit_button("🔍 Predict Insurance Cost")

    if submitted:
        input_data = pd.DataFrame({
            'age': [age],
            'sex': le_sex.transform([sex]),
            'bmi': [bmi],
            'children': [children],
            'smoker': le_smoker.transform([smoker]),
            'region': le_region.transform([region])
        })

        prediction = model.predict(input_data)[0]
        st.markdown(f"""
        <div class="alert alert-success mt-4" role="alert">
            💸 <strong>Estimated Insurance Cost:</strong> ${prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

