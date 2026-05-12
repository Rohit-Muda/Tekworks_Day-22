# Support Vector Machine (SVM)

## Types of SVM
- **SVC (Support Vector Classifier)** → Used for Classification problems
- **SVR (Support Vector Regressor)** → Used for Regression problems

---

# When SVM is Used

SVM is mainly used for:

- Classification Problems
- Binary Classification
- Non-linear Data
- High-dimensional datasets

### Real-world Examples
- Spam vs Not Spam
- Cat vs Dog
- Fraud vs Genuine Transaction
- Disease vs No Disease
- Positive vs Negative Review
- Face Recognition
- Text Classification

---

# Logistic Regression vs SVM

## Logistic Regression
- Works well when data is linearly separable
- Creates a linear decision boundary

### Problems with Logistic Regression
- Performs poorly on complex datasets
- Cannot handle non-linear boundaries effectively
- Struggles when classes overlap heavily
- Limited flexibility for complex patterns

---

# Why SVM is Preferred

SVM handles:
- Complex data distributions
- Non-linear relationships
- Overlapping classes
- High-dimensional feature spaces

It creates the **maximum margin boundary**, leading to better generalization.

---

# Important SVM Concepts

## 1. Hyperplane
A hyperplane is the line (2D) or plane (higher dimensions) that separates classes.

### Equation

```python
y = mx + c
```

In higher dimensions:

```python
w.x + b = 0
```

Where:
- `w` → weights
- `x` → input features
- `b` → bias

---

## 2. Decision Boundary
The final separating boundary between classes.

SVM tries to maximize the distance between:
- decision boundary
- nearest data points

---

## 3. Support Vectors
Support vectors are:
- the nearest data points to the decision boundary
- the most important points in SVM

They directly influence the position of the hyperplane.

---

## 4. Margin
The distance between:
- support vectors
- decision boundary

SVM aims to maximize this margin.

### Large Margin
- Better generalization
- Lower overfitting

---

# Kernels in SVM

Kernels help SVM handle non-linear data.

## Types of Kernels

### 1. Linear Kernel
Used when data is linearly separable.

```python
kernel='linear'
```

---

### 2. Polynomial Kernel
Creates curved boundaries.

```python
kernel='poly'
```

---

### 3. RBF Kernel (Most Common)
Handles highly complex and non-linear data.

```python
kernel='rbf'
```

---

### 4. Sigmoid Kernel
Works similar to neural networks.

```python
kernel='sigmoid'
```

---

# Parameters vs Hyperparameters

## Parameters
Values learned automatically from data during training.

### Examples
- weights
- bias

---

## Hyperparameters
Values set manually before training.

### Examples
- `C`
- `kernel`
- `gamma`
- `degree`

Hyperparameters control model behavior.

---

# Important SVM Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| C | Controls margin vs misclassification |
| kernel | Type of decision boundary |
| gamma | Influence of individual points |
| degree | Degree of polynomial kernel |

---

# Hyperparameter Tuning

## 1. GridSearchCV
- Tries every possible combination
- More accurate
- Computationally expensive

### Example

```python
from sklearn.model_selection import GridSearchCV
```

---

## 2. RandomizedSearchCV
- Tests random parameter combinations
- Faster than GridSearchCV
- Efficient for large search spaces

### Example

```python
from sklearn.model_selection import RandomizedSearchCV
```

---

# Advantages of SVM

- Effective in high-dimensional spaces
- Works well with small and medium datasets
- Handles non-linear data
- Memory efficient
- Strong mathematical foundation

---

# Disadvantages of SVM

- Slow on very large datasets
- Sensitive to feature scaling
- Choosing proper kernel is difficult
- Harder to interpret compared to linear models

---

# Feature Scaling in SVM

Feature scaling is extremely important because SVM is distance-based.

Use:

```python
from sklearn.preprocessing import StandardScaler
```

Without scaling:
- performance drops
- model becomes unstable

---

# Basic SVC Example

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

---

# Basic SVR Example

```python
from sklearn.svm import SVR

model = SVR(kernel='rbf')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

---

# Evaluation Metrics

## For Classification (SVC)
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## For Regression (SVR)
- R² Score
- MAE
- MSE
- RMSE

---

# Key Points

- SVM maximizes margin between classes
- Support vectors define the decision boundary
- Kernels transform non-linear data into separable space
- Scaling is mandatory for SVM
- RBF is the most commonly used kernel
- SVC is for classification, SVR is for regression
