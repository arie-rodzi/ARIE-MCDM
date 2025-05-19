
import streamlit as st
import pandas as pd
import numpy as np
import io

def normalize_matrix(X, types, targets):
    norm = []
    for j, t in enumerate(types):
        col = X[:, j].astype(float)
        if t == 'benefit':
            norm.append(col / np.max(col))
        elif t == 'cost':
            norm.append(np.min(col) / col)
        elif t == 'target':
            xT = float(targets[j])
            denom = max(abs(np.max(col) - xT), abs(np.min(col) - xT))
            norm.append(1 - abs(col - xT) / denom)
    return np.array(norm).T

def apply_weights(R, weights):
    return R * weights

def compute_similarity(V, gamma=1):
    v_max = V.max(axis=0)
    v_min = V.min(axis=0)
    S_best = np.sum((V / v_max) ** gamma, axis=1)
    S_worst = np.sum((v_min / V) ** gamma, axis=1)
    return S_best, S_worst

def compute_RC(S_best, S_worst, kappa=0.5):
    return (kappa * S_best) / (kappa * S_best + (1 - kappa) * S_worst)

st.set_page_config(page_title="ARIE - Improved Version", layout="wide")
st.title("ARIE - Adaptive Ranking with Ideal Evaluation (Improved Version)")

uploaded_file = st.file_uploader("Upload Improved Excel File", type=['xlsx'])

if uploaded_file:
    try:
        df_dict = pd.read_excel(uploaded_file, sheet_name=None)
        st.success(f"Sheets detected: {list(df_dict.keys())}")

        dm = df_dict["DecisionMatrix"]
        criteria = df_dict["CriteriaInfo"]
        params = df_dict["Parameters"]

        X = dm.iloc[:, 1:].values
        alternatives = dm.iloc[:, 0].values
        criteria_names = dm.columns[1:]

        # Map types, weights, targets
        types = criteria["Type"].str.lower().values
        weights = criteria["Weight"].astype(float).values
        targets = criteria["Target"].fillna(np.nan).values

        gamma = float(params["Gamma"].iloc[0])
        kappa = float(params["Kappa"].iloc[0])

        # Normalize & weight
        R = normalize_matrix(X, types, targets)
        V = apply_weights(R, weights)
        S_best, S_worst = compute_similarity(V, gamma)
        RC = compute_RC(S_best, S_worst, kappa)

        results_df = pd.DataFrame({
            "Alternative": alternatives,
            "RC Score": RC,
            "Rank": RC.argsort()[::-1].argsort() + 1
        }).sort_values("Rank")

        st.subheader("Ranking Results")
        st.dataframe(results_df)
        st.bar_chart(results_df.set_index("Alternative")["RC Score"])

        st.subheader("Parameters Used")
        st.write(f"Gamma (γ): {gamma}")
        st.write(f"Kappa (κ): {kappa}")

        st.subheader("Criteria Information")
        st.dataframe(criteria)

        # Export full Excel report
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(X, columns=criteria_names).to_excel(writer, sheet_name="Raw Data", index=False)
            pd.DataFrame(R, columns=criteria_names).to_excel(writer, sheet_name="Normalized", index=False)
            pd.DataFrame(V, columns=criteria_names).to_excel(writer, sheet_name="Weighted", index=False)
            results_df.to_excel(writer, sheet_name="Final Ranking", index=False)

        st.download_button("Download Excel Report", output.getvalue(), "ARIE_Report.xlsx")

    except Exception as e:
        st.error(f"Error processing file: {e}")
