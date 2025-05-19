# ARIE-MCDM Web App

**ARIE (Adaptive Ranking with Ideal Evaluation)** is a Multi-Criteria Decision-Making (MCDM) method designed to evaluate and rank alternatives based on benefit, cost, and target-type criteria. This Streamlit app provides an interactive way to upload input data, compute ARIE scores, visualize rankings, and download results.

---

## 🚀 Live Demo

You can deploy this app using [Streamlit Cloud](https://streamlit.io/cloud) by linking to this GitHub repository.

---

## 📁 Project Structure

```
📦 arie-mcdm-app/
├── app_arie_streamlit.py          # Streamlit web app
├── requirements.txt               # Required Python libraries
├── ARIE_MCDM_Input_Template.xlsx  # Sample input Excel template
└── README.md                      # Project documentation
```

---

## 📥 Input Format (Excel)

Your input Excel file must have **three sheets**:

### 1. `DecisionMatrix`
| Alternative | C1  | C2  | C3  | C4  |
|-------------|-----|-----|-----|-----|
| A1          | ... | ... | ... | ... |

### 2. `CriteriaInfo`
| Criterion | Type    | Weight | Value |
|-----------|---------|--------|--------|
| C1        | Benefit | 0.25   | none   |
| C2        | Cost    | 0.25   | none   |
| C3        | Target  | 0.25   | 12     |
| C4        | Benefit | 0.25   | none   |

- Types: `Benefit`, `Cost`, or `Target`
- Use `"none"` for value if not Target

### 3. `Parameters`
| Parameter | Value |
|-----------|--------|
| alpha     | 0.5    |
| lambda    | 0.3    |

---

## ✅ Features

- Handles 3 types of criteria: Benefit, Cost, and Target
- Auto-normalizes data based on type
- Displays:
  - Original Matrix
  - Normalized Matrix
  - Weighted Matrix
  - Final Scores + Rankings
  - Bar Chart of Scores
- Download all results (Excel + Chart) in one ZIP

---

## 🛠 How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app_arie_streamlit.py
```

---

## ☁️ How to Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Connect to your repo and select `app_arie_streamlit.py`
5. Click **Deploy**

---

## 📧 Contact

Developed by Dr. Zahari Md Rodzi  
Email: [zaharimdrodzi@gmail.com](mailto:zaharimdrodzi@gmail.com)
