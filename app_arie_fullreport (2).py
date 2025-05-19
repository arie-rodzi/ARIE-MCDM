
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="ARIE - Full Report", layout="wide")
st.title("ARIE - Adaptive Ranking with Ideal Evaluation (Full Report)")

# Upload input Excel file
uploaded_file = st.file_uploader("Upload the Improved Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Read all sheets from the uploaded Excel
        df_decision = pd.read_excel(uploaded_file, sheet_name="DecisionMatrix", index_col=0)
        df_weights = pd.read_excel(uploaded_file, sheet_name="Weights", header=None)
        df_types = pd.read_excel(uploaded_file, sheet_name="Types", header=None)
        df_parameters = pd.read_excel(uploaded_file, sheet_name="Parameters")
        df_targets = pd.read_excel(uploaded_file, sheet_name="Targets", header=None)

        alternatives = df_decision.index.tolist()
        criteria_names = df_decision.columns.tolist()
        weights = df_weights.values[0].astype(float)
        types = df_types.values[0].astype(str)
        targets = df_targets.values[0].astype(float)
        gamma = float(df_parameters.loc[0, "Gamma"])
        kappa = float(df_parameters.loc[0, "Kappa"])

        st.subheader("Raw Decision Matrix")
        st.dataframe(df_decision)

        st.subheader("Criteria Info")
        df_criteria = pd.DataFrame({
            "Criterion": criteria_names,
            "Type": types,
            "Weight": weights,
            "Target": targets
        })
        st.dataframe(df_criteria)

        norm_matrix = []
        for j, col in enumerate(criteria_names):
            vals = df_decision[col].values.astype(float)
            if types[j].lower() == "benefit":
                norm = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
            elif types[j].lower() == "cost":
                norm = (np.max(vals) - vals) / (np.max(vals) - np.min(vals))
            elif types[j].lower() == "target":
                target_val = targets[j]
                diff = np.abs(vals - target_val)
                norm = 1 - diff / np.max(diff) if np.max(diff) != 0 else np.ones_like(vals)
            else:
                norm = vals
            norm_matrix.append(norm)
        norm_matrix = np.array(norm_matrix).T

        df_norm = pd.DataFrame(norm_matrix, index=alternatives, columns=criteria_names)
        st.subheader("Normalized Matrix")
        st.dataframe(df_norm)

        weighted_matrix = norm_matrix * weights
        df_weighted = pd.DataFrame(weighted_matrix, index=alternatives, columns=criteria_names)
        st.subheader("Weighted Normalized Matrix")
        st.dataframe(df_weighted)

        rc_scores = gamma * np.mean(weighted_matrix, axis=1) + kappa * np.std(weighted_matrix, axis=1)
        rc_ranks = np.argsort(-rc_scores) + 1
        df_results = pd.DataFrame({
            "Alternative": alternatives,
            "RC Score": rc_scores,
            "Rank": rc_ranks
        })
        df_results = df_results.sort_values("Rank").reset_index(drop=True)
        st.subheader("Final RC Ranking Table")
        st.dataframe(df_results.style.format({"RC Score": "{:.6f}"}))

        st.subheader("Bar Chart of RC Scores (Top 30 Alternatives)")
        top_n = 30
        df_chart = df_results.head(top_n)
        fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 2))
        ax.barh(df_chart["Alternative"], df_chart["RC Score"], color="skyblue")
        ax.set_xlabel("RC Score")
        ax.set_ylabel("Alternative")
        ax.set_title("RC Score by Alternative (Top 30)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_decision.to_excel(writer, sheet_name="Raw Decision Matrix")
            df_criteria.to_excel(writer, sheet_name="Criteria Info", index=False)
            df_norm.to_excel(writer, sheet_name="Normalized Matrix")
            df_weighted.to_excel(writer, sheet_name="Weighted Matrix")
            df_results.to_excel(writer, sheet_name="Final Ranking", index=False)
        output.seek(0)
        st.subheader("Download Full Excel Report")
        st.download_button("📥 Download Excel Report", data=output, file_name="ARIE_Full_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Error: {e}")
