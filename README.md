# 🗳️ Structural Democracy Scoring System

## 📌 Overview

The Structural Democracy Scoring System is a machine-learning-powered application that classifies countries as **Democratic** or **Non-Democratic** based on real-world structural indicators. Unlike conventional indices that reflect geopolitical alignment or ideological expectations, this system focuses on **internal legitimacy**, **behavioral civic engagement**, and the **public’s anticipation of their future within the system**.

---

## 🎯 Purpose

This project is designed to measure democracy not by international standards, but by one core principle:

> **A democratic system is one in which the people act today as if the future is theirs.**

This system answers:
- Can the public influence and replace leadership?
- Can they advocate, organize, and prosper regardless of status?
- Do people act — not just hope — toward a shared future?

---

## 🧠 Core Philosophy

The model is based on three foundational ideas:

1. **Humans are future-driven:** their sense of legitimacy depends on anticipated possibilities, not just current material conditions.
2. **Agency must be activated:** democracy exists only when people are already using their voice, not merely allowed to.
3. **Democracy is structurally resilient:** legitimacy must be sustained across time, conflict, and dissent.

---

## 🧱 System Architecture

### 🔹 Input:
- A tabular dataset (CSV or Excel) with 20 democracy indicators across 5 domains.
- Each country is assigned scores from **0–10** per feature, with a final label: `'Democratic'` or `'Non-Democratic'`.

### 🔹 Processing:
- Data cleaning, scaling, and train/test splitting
- Classification using a Random Forest model
- Optional hyperparameter tuning with GridSearch or RandomSearch
- Cross-validation for robustness

### 🔹 Output:
- Textual evaluation: accuracy, confusion matrix, classification report
- Printed model parameters and validation scores

---

## 🧩 Domains & Features

| Domain | Features |
|--------|----------|
| **1. Power Responsiveness** | Leadership Turnover, Local Governance Autonomy, Policy Responsiveness |
| **2. Sociopolitical Coherence** | Language Alignment, Participation Accessibility, Civilian Control, Structural Advocacy & Prosperity (for minorities) |
| **3. Material Well-being** | Healthcare Access, Education Completion, Food Sovereignty, Migration Freedom |
| **4. Future Perception** | Agency Projection, Pathway to Influence, Collective Future Confidence, System Adaptability |
| **5. Behavioral Engagement** | Collective Momentum, Life Investment, Exit vs. Voice Behavior, Narrative-Action Alignment |

Each score must reflect observable public behavior or systemic accessibility — not ideology or sentiment.

---

## 🛠️ How It Works

1. **Prepare Data**  
   - Load structured data using the provided schema  
   - Drop missing entries  
   - Separate features (`X`) and labels (`y`)

2. **Train Model**  
   - Standardize features  
   - Fit a RandomForestClassifier  
   - Predict and evaluate performance on test data

3. **Validation & Optimization**  
   - Run k-fold cross-validation (default 10-fold)  
   - (Optional) Run GridSearchCV or RandomizedSearchCV  
   - Print accuracy, best parameters, and diagnostics

---

## 🔍 Why This Is Different

- ✅ **Non-ideological**: Not reliant on institutions whose existence depends on global conformity
- ✅ **Ground-truth based**: Derived from action and structural access, not sentiment
- ✅ **Future-oriented**: Captures how people *behave toward tomorrow*, not just what laws exist today
- ✅ **Lean & reproducible**: No complex dependencies, no ideological gatekeeping

---

## 📁 Example Use Cases

- Classify 20–30 countries for a policy research report  
- Generate regional democratic risk maps based on behavioral data  
- Compare elite-imposed vs. people-driven democratic systems  
- Anticipate instability based on divergence between structure and popular perception

---

## 🛤️ Extensibility

This project is built for expansion:
- Add temporal dimension to track democratic trajectory
- Visualize regime drift or recovery based on score shifts
- Simulate downstream effects of structural changes

---

## 🧾 Credits & Licensing

This tool is developed as an open framework for structurally measuring legitimacy without ideological dependencies. It is intended for public use, education, and policy refinement.

---
