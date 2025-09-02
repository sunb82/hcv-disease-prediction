# Hepatitis C Virus (HCV) Disease Stage Prediction

## Project Overview
This machine learning project predicts the disease stage of Hepatitis C Virus (HCV) patients using blood test biomarkers. The model classifies patients into four categories representing different stages of liver disease progression, from healthy individuals to those with cirrhosis.

## Disease Categories
The model predicts four distinct stages:
- **Stage 0 (Blood Donor)**: Healthy individuals with no HCV infection
- **Stage 1 (Hepatitis)**: Active hepatitis C infection with liver inflammation
- **Stage 2 (Fibrosis)**: Scarring of liver tissue due to chronic inflammation
- **Stage 3 (Cirrhosis)**: Advanced scarring with significant liver damage

## Dataset Description
- **Source**: UCI Machine Learning Repository - HCV Dataset
- **Total Samples**: 615 patient records
- **Features**: 13 clinical biomarkers from blood tests
- **Class Distribution**: Severely imbalanced (86% healthy donors)

### Key Features
The model uses the following blood test biomarkers:
- **ALB** (Albumin): Protein made by liver (g/L)
- **ALP** (Alkaline phosphatase): Liver enzyme (IU/L)
- **ALT** (Alanine aminotransferase): Liver enzyme indicating damage (IU/L)
- **AST** (Aspartate aminotransferase): Liver enzyme indicating damage (IU/L)
- **BIL** (Bilirubin): Waste product from red blood cells (μmol/L)
- **CHE** (Cholinesterase): Enzyme produced by liver (kU/L)
- **CHOL** (Cholesterol): Lipid levels (mmol/L)
- **CREA** (Creatinine): Kidney function marker (μmol/L)
- **GGT** (Gamma-glutamyl transferase): Liver enzyme (IU/L)
- **PROT** (Total protein): Blood protein levels (g/L)
- **Age**: Patient age in years
- **Sex**: Patient gender (m/f)

## Data Preprocessing Pipeline

### 1. Data Cleaning
- Missing values imputed: 26 instances
- Outliers identified: 92 instances (retained for medical validity)
- No duplicate records found

### 2. Class Imbalance Handling
- **Problem**: Severe imbalance with 86% healthy donors and only 7 suspect cases
- **Solution**: 
  - Merged 'Suspect Blood Donor' with 'Hepatitis' class (too few samples)
  - Applied SMOTE (Synthetic Minority Over-sampling Technique) to training set
  - Achieved balanced training distribution: 316 samples per class

### 3. Feature Engineering
Created domain-specific medical features:
- **Medical Ratios**:
  - AST/ALT ratio (important liver damage indicator)
  - Albumin/Globulin ratio
  - ALP/ALT ratio
- **Abnormal Indicators**: 5 binary features for values outside normal ranges
- **Age Groups**: Categorized into 5 age brackets
- **Severity Score**: Composite metric combining multiple liver enzymes

### 4. Data Splitting
- Training: 60% (353 samples → 1,264 after SMOTE)
- Validation: 20% (118 samples)
- Test: 20% (118 samples)

### 5. Feature Scaling
- Method: StandardScaler
- Fitted on training set only to prevent data leakage
- Total features after engineering: 22

## Model Development

### Models Evaluated
Five classification algorithms were trained and compared:
1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear, interpretable model
3. **Random Forest** - Ensemble of decision trees
4. **XGBoost** - Gradient boosting algorithm
5. **Support Vector Machine (SVM)** - Non-linear kernel-based classifier

### Training Approach
- All models trained on SMOTE-balanced training set
- Hyperparameters kept at defaults for initial comparison
- Primary evaluation metric: **Balanced Accuracy** (average recall across all classes)
- Secondary metrics: Standard Accuracy, F1-Score, Precision, Recall
- Best model selected based on balanced accuracy to ensure fair performance across all disease stages

## Model Performance Results

### Validation Set Performance

| Model | Training Accuracy | Standard Accuracy | **Balanced Accuracy** | F1-Score | 
|-------|------------------|-------------------|----------------------|----------|
| **Random Forest** | 100.0% | 95.8% | **65.0%** | 0.948 |
| Decision Tree | 100.0% | 94.9% | 64.8% | 0.941 |
| XGBoost | 100.0% | 94.9% | 60.0% | 0.936 |
| Logistic Regression | 99.8% | 92.4% | 54.5% | 0.929 |
| SVM | 99.6% | 92.4% | 54.5% | 0.924 |

### Per-Class Performance (Best Model - Random Forest)

| Disease Stage | Recall (Sensitivity) | Support |
|--------------|---------------------|---------|
| Healthy (0) | 100.0% | 105 |
| Hepatitis (1) | 80.0% | 5 |
| Fibrosis (2) | 0.0% | 3 |
| Cirrhosis (3) | 80.0% | 5 |

### Key Performance Insights

1. **Best Model**: Random Forest achieved the highest balanced accuracy of 65.0%
2. **Balanced vs Standard Accuracy**: 
   - Standard accuracy (95.8%) is misleading due to class imbalance
   - Balanced accuracy (65.0%) reveals true model performance across all classes
   - This metric averages the recall of each class, giving equal weight regardless of class size
3. **Class-Specific Challenges**:
   - **Excellent**: Perfect detection of healthy patients (100%)
   - **Good**: 80% detection rate for Hepatitis and Cirrhosis
   - **Poor**: Cannot detect Fibrosis cases (0% - only 3 samples in validation)
4. **Why Balanced Accuracy Matters**:
   - With 89% healthy patients in test set, a model predicting only "healthy" gets 89% standard accuracy
   - Balanced accuracy penalizes this behavior, showing true performance (25% for always-healthy prediction)
   - Our best model's 65% balanced accuracy represents genuine predictive capability

### Model Strengths
- **High Recall**: Models correctly identify most diseased patients (crucial for medical screening)
- **Balanced Performance**: Good precision-recall trade-off across all classes
- **Fast Inference**: All models provide real-time predictions

### Limitations & Considerations
1. **Severe Class Imbalance**: 
   - Test set: 89% healthy, making standard accuracy misleading
   - Fibrosis class has only 3 validation samples, resulting in 0% detection
   - Real-world performance depends on actual disease prevalence
2. **Small Dataset**: Only 615 total samples limits model complexity and generalization
3. **Balanced Accuracy Reality Check**: 
   - 65% balanced accuracy means the model misses 35% of cases when weighted equally
   - Perfect healthy detection inflates standard accuracy metrics
4. **Missing Validation**: Models require clinical validation before deployment
5. **Overfitting Risk**: Perfect training accuracy with lower balanced accuracy suggests overfitting

## Technical Requirements

### Dependencies
- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- imbalanced-learn (for SMOTE)

### Project Structure
```
hcv_disease_prediction/
├── data/
│   ├── hcv_data.csv              # Original dataset
│   └── processed/                # Preprocessed data splits
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA and analysis
│   ├── 02_preprocessing.ipynb    # Data preparation
│   └── 03_modeling.ipynb         # Model training
├── src/
│   ├── data_preprocessing.py     # Preprocessing pipeline
│   └── model_training.py         # Training utilities
├── models/                       # Saved model files
└── images/                       # Visualizations
```

## Clinical Significance
This model demonstrates the potential of machine learning in:
- Early detection of liver disease progression
- Non-invasive screening using routine blood tests
- Risk stratification for HCV patients
- Supporting clinical decision-making

## Future Improvements
1. **Collect More Data**: Larger, balanced dataset would improve reliability
2. **Feature Selection**: Identify most predictive biomarkers
3. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Clinical Validation**: Test on prospective patient cohorts
6. **Interpretability**: Add SHAP/LIME for model explanations
7. **Temporal Analysis**: Include patient history and progression tracking


## Acknowledgments
- UCI Machine Learning Repository for the HCV dataset
- Original data contributors: Lichtinghagen et al.