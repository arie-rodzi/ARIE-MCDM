
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIE - Full Report", layout="wide")
st.title("ARIE - Adaptive Ranking with Ideal Evaluation")

# Upload input Excel
uploaded_file = st.file_uploader("Upload Improved Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df_input = pd.read_excel(uploaded_file, sheet_name=None)
        decision_matrix = df_input['DecisionMatrix']
        weights = df_input['Weights'].iloc[0].values
        types = df_input['Types'].iloc[0].values
        gamma = float(df_input['Parameters'].loc[0, 'Gamma'])
        kappa = float(df_input['Parameters'].loc[0, 'Kappa'])

        st.subheader("Raw Decision Matrix")
        st.dataframe(decision_matrix)

        # Step 1: Normalize
        norm_matrix = []
        for j, col in enumerate(decision_matrix.columns):
            vals = decision_matrix[col].values
            if types[j] == "benefit":
                norm = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
            elif types[j] == "cost":
                norm = (np.max(vals) - vals) / (np.max(vals) - np.min(vals))
            else:  # target
                target_val = float(df_input['Targets'].iloc[0, j])
                norm = 1 - np.abs(vals - target_val) / np.max(np.abs(vals - target_val))
            norm_matrix.append(norm)
        norm_matrix = np.array(norm_matrix).T

        # Step 2: Apply Weights
        weighted = norm_matrix * weights

        # Step 3: Calculate RC Score
        rc_scores = gamma * np.mean(weighted, axis=1) + kappa * np.std(weighted, axis=1)
        rc_ranks = np.argsort(-rc_scores) + 1

        df_result = pd.DataFrame({
            "Alternative": decision_matrix.index,
            "RC Score": rc_scores,
            "Rank": rc_ranks
        }).sort_values("Rank").reset_index(drop=True)

        st.subheader("Final RC Ranking Table")
        st.dataframe(df_result.style.format({"RC Score": "{:.6f}"}))

        # Step 4: Bar Chart
        st.subheader("Bar Chart of RC Scores")
        fig, ax = plt.subplots(figsize=(10, 20))
        ax.barh(df_result["Alternative"], df_result["RC Score"], color="skyblue")
        ax.set_xlabel("RC Score")
        ax.set_ylabel("Alternative")
        ax.set_title("RC Score by Alternative")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Step 5: Download
        st.subheader("Download Full Results")
        def convert_df(df):
            return df.to_excel(index=False, engine='openpyxl')

        st.download_button(
            "📥 Download Excel Report",
            convert_df(df_result),
            "ARIE_Full_Report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
