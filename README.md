# Developer Salary & Job Satisfaction Analysis  
📊 **Data Science Project — Stack Overflow Developer Survey**

This project analyzes global developer compensation and job satisfaction using the Stack Overflow Developer Survey.  
The goal is to identify which factors significantly influence annual salary and the likelihood of high job satisfaction (JobSat ≥ 8).

The analysis follows a CRISP-DM workflow and includes data cleaning, feature engineering, EDA, statistical tests, and machine learning models.

---

## 🎯 Objectives

This study investigates four central questions:

1. **What is the expected annual salary given a developer’s country, years of experience, and technology stack?**
2. **Does industry affect salary and satisfaction independently of experience?**
3. **Is remote work associated with higher or lower salaries when controlling for country and experience?**
4. **Is high job satisfaction (JobSat ≥ 8) influenced by remote work, management role, and experience level, after controlling for country and industry?**

---

## 🛠️ Technologies Used

- **Python 3**
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- statsmodels, scipy  
- Jupyter Notebook  

---

## 📁 Project Structure

```
dev-salary-satisfaction-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                              # Dataset folder (survey.csv NOT committed)
│
├── notebook/
│   └── data-science-project-dev-salary-satisfaction-analysis.ipynb
│
├── src/                               # Modularized code (optional for scaling project)
│   ├── preprocessing.py
│   ├── eda.py
│   ├── modeling_salary.py
│   ├── modeling_jobsat.py
│   └── utils.py
│
└── results/
    ├── figures/                        # Exported visualizations
    ├── model_outputs/                  # Metrics, coefficients, confusion matrices
    └── tables/                         # Encoded features, cleaned data summaries
```

---

## 📂 Dataset Information

The original **survey.csv** file (≈134 MB) is **not included** in this repository due to GitHub's file size limits.

### 📥 How to Get the Data

1. Download the dataset from **[Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025/)**.  
2. Place the file inside the `data/` folder as:

```
data/survey.csv
```

3. The notebook will automatically load it using a **relative path**, ensuring portability:

```python
data_path = "../data/survey.csv"
```

---

## 🚀 How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/yourusername/dev-salary-satisfaction-analysis.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download `survey.csv` and place it inside `/data`.

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```
notebook/data-science-project-dev-salary-satisfaction-analysis.ipynb
```

---

## 🔍 Methodology

### **1. Data Cleaning**
- Handling missing values  
- Standardizing survey responses  
- Removing and documenting outliers  
- Log-transforming salary variables  

### **2. Feature Engineering**
- **Target encoding**: Country  
- **One-Hot encoding**: Industry, DevType  
- **Binary encoding**: RemoteWork categories  
- **Ordinal encoding**: Education Level, Organization Size  
- Experience log transformation  

### **3. Exploratory Data Analysis**
- Salary distribution and country differences  
- Experience vs. compensation trends  
- Technology stack frequency  
- Satisfaction patterns  

### **4. Modeling**
- Multiple linear regression  
- Ridge and Lasso regularization  
- Random Forest Regressor  
- Welch’s ANOVA for industry comparisons  
- Logistic regression and Random Forest for job satisfaction  

### **5. Evaluation**
- RMSE, MAE, R²  
- Confusion matrix, accuracy, precision, recall  
- Feature importance and coefficient interpretation  

---

## 📈 Key Results

- **Country is the strongest driver of salary**, even when controlling for experience.  
- **Experience increases salary but at a decreasing rate** after several years.  
- **Industry significantly affects salary** (Finance, AI, Cloud highest; Education and Government lowest).  
- **Remote work is associated with slightly higher salaries** when controlling for country and experience.  
- **High job satisfaction** is most strongly linked to:  
  - Remote/flexible work  
  - Being a manager/lead  
  - More experience  
  - Certain industries with better work environments  

Full details are available in the notebook.

---

## 📊 Dataset Characteristics

- ~90,000 developer responses  
- 170+ survey columns  
- Global coverage across 180+ countries  
- Includes demographics, experience, industry, salary, job satisfaction, and technical stack

---

## 👤 Author

**Camila Fonseca**  
🔗 LinkedIn: https://www.linkedin.com/in/camila-fonseca/

---

## 📄 License

This project is open source under the **MIT License**.
