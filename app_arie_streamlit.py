
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIE MCDM", layout="wide")
st.title("ARIE (Adaptive Ranking with Ideal Evaluation) Method")

def normalize_matrix(df, criteria_info):
    norm_df = df.copy()
    targets = {row["Criterion"]: row["Value"] for _, row in criteria_info.iterrows() if row["Type"] == "Target"}
    for _, row in criteria_info.iterrows():
        col = row["Criterion"]
        typ = row["Type"]
        if typ == "Benefit":
            norm_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif typ == "Cost":
            norm_df[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
        elif typ == "Target":
            T = float(targets[col])
            norm_df[col] = 1 - abs(df[col] - T) / abs(df[col] - T).max()
    return norm_df

uploaded_file = st.file_uploader("Upload Excel file with sheets: DecisionMatrix, CriteriaInfo, Parameters", type=["xlsx"])

if uploaded_file:
    try:
        decision_df = pd.read_excel(uploaded_file, sheet_name="DecisionMatrix")
        criteria_info = pd.read_excel(uploaded_file, sheet_name="CriteriaInfo")
        parameters = pd.read_excel(uploaded_file, sheet_name="Parameters")
        params_dict = dict(zip(parameters["Parameter"], parameters["Value"]))
        alpha = params_dict.get("alpha", 0.5)
        lambda_ = params_dict.get("lambda", 0.5)

        st.sidebar.subheader("Parameters")
        st.sidebar.write(params_dict)

        criteria = criteria_info["Criterion"].tolist()
        weights = criteria_info["Weight"].values

        st.subheader("Step 1: Original Decision Matrix")
        st.dataframe(decision_df)

        norm_matrix = normalize_matrix(decision_df[criteria], criteria_info)
        norm_matrix.insert(0, "Alternative", decision_df["Alternative"])

        st.subheader("Step 2: Normalized Decision Matrix")
        st.dataframe(norm_matrix)

        weighted_matrix = norm_matrix.copy()
        for col, w in zip(criteria, weights):
            weighted_matrix[col] = weighted_matrix[col] * w

        st.subheader("Step 3: Weighted Normalized Matrix")
        st.dataframe(weighted_matrix)

        scores = norm_matrix[criteria].dot(weights)
        decision_df["Score"] = scores
        decision_df["Rank"] = decision_df["Score"].rank(ascending=False).astype(int)

        st.subheader("Step 4: Final Scores and Ranking")
        st.dataframe(decision_df[["Alternative", "Score", "Rank"]])

        st.subheader("Step 5: Ranking Bar Chart")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(decision_df["Alternative"], decision_df["Score"], color='skyblue')
        ax.set_xlabel("Alternative")
        ax.set_ylabel("ARIE Score")
        ax.set_title("ARIE Method Ranking")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            decision_df.to_excel(writer, sheet_name="FinalRanking", index=False)
            norm_matrix.to_excel(writer, sheet_name="NormalizedMatrix", index=False)
            weighted_matrix.to_excel(writer, sheet_name="WeightedMatrix", index=False)
            parameters.to_excel(writer, sheet_name="Parameters", index=False)

        st.download_button(
            label="📥 Download Result as Excel",
            data=output.getvalue(),
            file_name="ARIE_MCDM_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
