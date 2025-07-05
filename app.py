import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ✅ Load model and scaler
with open("parkinsons_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Define feature names
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
    'spread1', 'spread2', 'D2', 'PPE'
]

# ✅ Streamlit App UI
st.title("🧠 Parkinson's Disease Prediction App")
st.markdown("Enter the voice measurements below to predict the likelihood of Parkinson’s Disease.")

# ✅ Input fields
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}:", format="%.6f", key=feature)
    user_input.append(val)

# ✅ Predict Button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict_proba(input_scaled)[0][1]

    # 🔁 Custom threshold logic
    if probability > 0.7:
        prediction = 1
    else:
        prediction = 0

    # ✅ Show prediction and explanation
    st.metric("Confidence", f"{probability:.2%}")
    st.progress(probability)

    if prediction == 1:
        st.error(f"🔴 The model predicts **Parkinson’s Disease** with {probability:.2%} confidence.")
        with st.expander("📘 What does this mean?"):
            st.write("""
            This result suggests that the vocal patterns you entered are **similar to those found in individuals with Parkinson's Disease**.

            👉 Please note that this is **not a diagnosis**.  
            It is a machine learning prediction based on your voice measurements.  
            You should consult a neurologist or doctor for clinical evaluation.
            """)
    else:
        st.success(f"🟢 The model predicts **Healthy** with {(1 - probability):.2%} confidence.")
        with st.expander("📘 What does this mean?"):
            st.write("""
            The model finds your voice measurement values to be **similar to those of healthy individuals**.

            ✅ That’s a good sign! However, if you're experiencing symptoms or are at risk,  
            please consider consulting a doctor for further screening.

            This app is meant for educational and screening purposes only.
            """)

# ✅ Feature Importance Visualization
st.subheader("🔍 How the model makes decisions")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
st.caption("This chart shows which voice features most influenced the prediction.")
