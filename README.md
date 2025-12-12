Task 1 — House Price Prediction (Linear Regression)

This task focuses on building a complete end-to-end Machine Learning pipeline to predict house prices using the **Ames Housing Dataset**.  
The project covers data understanding, preprocessing, feature engineering, model training, evaluation, and result visualization.

---

## 📌 Objective
Develop a regression model to predict property sale prices based on various housing features such as:
- Area
- Number of rooms
- Location
- Quality and condition
- Basement, garage, and exterior characteristics

---

## 📂 Task Components

### **1️⃣ Data Loading**
- Loaded the Ames Housing dataset into a pandas DataFrame.
- Inspected shape, column types, and target variable (`SalePrice`).

### **2️⃣ Exploratory Data Analysis (EDA)**
Performed detailed EDA including:
- Distribution plots for `SalePrice`
- Missing value analysis
- Correlation heatmap
- Scatter plots for top predictors
- Boxplots for categorical feature impacts  
- Outlier visualization using boxplots  
- Pairplots and jointplots for key relationships  

### **3️⃣ Data Preprocessing**
Cleaned and transformed the dataset using:
- Logical missing-value fills (e.g., `0` for no garage/basement)
- Categorical `None` fills for absent features
- Rare category grouping (<1%)
- Ordinal encoding (e.g., quality ratings)
- Skew correction using `log1p`
- Normalization and scaling of numeric features
- One-hot encoding of categorical variables

### **4️⃣ Feature Engineering**
Created meaningful new variables:
- `TotalBathrooms`
- `TotalBsmtFinished`
- `HouseAge`
- `Remodeled` flag  
- `QualityIndex`
- `PricePerSqft`

These enhanced model interpretability and performance.

### **5️⃣ Model Pipeline**
Built an end-to-end scikit-learn pipeline:
- ColumnTransformer (Numeric + Categorical)
- SelectKBest (feature selection)
- Linear Regression model

This ensures reproducible and clean ML workflow.

### **6️⃣ Model Training**
Trained the model using the log-transformed target (`SalePrice_log`) for stability.

### **7️⃣ Evaluation**
Metrics used:
- **R² score** (original scale)
- **RMSE** (original scale)
- **R² on log-transformed target**

Visual evaluations:
- Actual vs Predicted plot
- Residuals vs Predicted plot
- Residual distribution histogram

### **8️⃣ Saving Outputs**
Exported:
- Trained model pipeline (`.joblib`)
- Test predictions (CSV)
- Visual plots and diagnostics

---

## 🚀 How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies if required:
   ```bash
   pip install numpy
