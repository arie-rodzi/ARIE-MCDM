
# ARIE - Adaptive Ranking with Ideal Evaluation

**ARIE** is a novel Multi-Criteria Decision-Making (MCDM) method developed to handle benefit, cost, and target-type criteria with an adaptive scoring mechanism. This repository contains a demo web application built using Streamlit.

> ⚠️ The full technical formulation of ARIE is not disclosed in this repository until the method is formally published.

---

## 🚀 Features

- Accepts decision matrices with any combination of:
  - Benefit-type criteria
  - Cost-type criteria
  - Target-type criteria (with specific desired values)
- Supports up to 100+ alternatives
- Uses adjustable parameters:
  - **Gamma (γ)**: Sensitivity parameter
  - **Kappa (κ)**: Compromise balancing parameter
- Automatically displays and exports:
  - Normalized matrix
  - Weighted matrix
  - Similarity scores
  - Final ranking
  - Interactive chart
- One-click full Excel report download

---

## 📄 Excel Input Format

Prepare your file with 3 sheets:

1. **DecisionMatrix**
    - Rows: Alternatives
    - Columns: Criteria (C1, C2, ..., Cn)
2. **CriteriaInfo**
    - Columns: `Criterion`, `Type`, `Weight`, `Target`
    - Types: `benefit`, `cost`, or `target`
3. **Parameters**
    - Columns: `Gamma`, `Kappa`

Use our sample template: [ARIE_template_improved.xlsx](ARIE_template_improved.xlsx)

---

## 🧪 Local Setup

```bash
pip install streamlit pandas numpy openpyxl matplotlib
streamlit run app_arie_fullreport.py
```

---

## 🌐 Online Demo (Coming Soon)

A demo version will be hosted on Streamlit Cloud.

---

## 📜 License

This repository is released under the [BSD 3-Clause License](LICENSE).

---

## 📚 Citation (Coming Soon)

Please cite our forthcoming publication when using ARIE in academic or commercial settings.

> "Rodzi, Z.M., et al. (2025). ARIE: Adaptive Ranking with Ideal Evaluation for MCDM Problems. *Journal TBD*."

---

## 🔒 Disclosure

This repository includes only the application and interface logic. The full formulation and proof of the ARIE method will be released after peer-reviewed publication.

