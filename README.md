# 📍 K-Nearest Neighbors (KNN): A Practical Guide

> **To predict a new case, look at the examples most like it.**

## 🗺️ Trail Map

- [What Is KNN?](#-what-is-knn)
- [How KNN Works](#-how-knn-works)
- [Classification and Regression](#-classification-and-regression)
- [Distance Measures](#-distance-measures)
- [Choosing the Value of k](#-choosing-the-value-of-k)
- [Feature Scaling](#-feature-scaling)
- [Python Example: Classification](#-python-example-classification)
- [Python Example: Regression](#-python-example-regression)
- [Strengths and Limitations](#️-strengths-and-limitations)
- [When to Use KNN](#-when-to-use-knn)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔦 What Is KNN?

**K-Nearest Neighbors**, usually shortened to **KNN**, is a supervised learning method used for classification and regression.

KNN compares a new example with examples in the training data. It finds the `k` examples considered nearest according to a chosen distance measure, then uses their labels or values to make a prediction.

KNN is often called a **non-parametric**, **instance-based**, or **lazy-learning** method:

- **Non-parametric:** It does not assume a particular equation describes the data.
- **Instance-based:** It keeps the training examples and uses them to predict new cases.
- **Lazy learning:** Most of the work happens when a prediction is requested rather than during a separate model-fitting stage.

Despite its simple idea, KNN can be effective when similar examples tend to have similar outcomes. Scikit-learn provides KNN classifiers, regressors, and tools for finding neighbors. ([scikit-learn.org](https://scikit-learn.org/stable/modules/neighbors.html))

## 🔄 How KNN Works

For a new example, KNN generally:

1. Measures its distance from the training examples.
2. Finds the `k` closest examples.
3. Uses those neighbors to predict the outcome.

```text
New example
     ↓
Calculate distances to training examples
     ↓
Select the k nearest examples
     ↓
Vote on a class or average a value
     ↓
Return a prediction
```

Imagine a new fruit described by its weight and color. If its nearest neighbors are mostly apples, a KNN classifier may predict that it is an apple.

The prediction depends on how the data is represented, the distance measure, and the chosen value of `k`.

## 🏷️ Classification and Regression

### KNN classification

For classification, KNN assigns the new example the most common class among its nearest neighbors.

If `k = 5` and three neighbors belong to class `A` while two belong to class `B`, the model predicts class `A`.

Some implementations can also return class scores based on neighbor votes. These should not automatically be treated as calibrated probabilities.

### KNN regression

For regression, KNN predicts a numerical value based on the target values of the nearest neighbors. A common method is to average their values.

If the three nearest examples have target values `10`, `12`, and `14`, an unweighted KNN regressor predicts their average: `12`.

Scikit-learn describes `KNeighborsRegressor` as predicting through local interpolation of the targets associated with neighboring examples. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html))

## 📏 Distance Measures

A distance measure defines what “near” means. The right choice depends on the data and how its features are represented.

### Euclidean distance

Euclidean distance is the straight-line distance between two points. For two features:

```text
distance = √((x₁ - y₁)² + (x₂ - y₂)²)
```

It is a common choice for numerical data.

### Manhattan distance

Manhattan distance adds the absolute differences across features:

```text
distance = |x₁ - y₁| + |x₂ - y₂| + ... + |xₙ - yₙ|
```

It can be useful when movement or difference is better represented as a sum across dimensions.

### Minkowski distance

Minkowski distance is a general family that includes Euclidean and Manhattan distance as special cases. In scikit-learn, the `p` parameter controls which Minkowski distance is used:

- `p=1`: Manhattan distance
- `p=2`: Euclidean distance

### Hamming distance

Hamming distance counts positions that differ. It can be appropriate for certain categorical or binary representations, but the features need to be encoded and interpreted carefully.

A distance metric is not automatically meaningful just because it can be calculated. The feature representation and domain matter.

## 🎛️ Choosing the Value of `k`

The value of `k` is the number of neighbors used to make a prediction.

- **Small `k`:** The model is strongly influenced by nearby examples. It can capture local detail, but may be sensitive to noise.
- **Large `k`:** The prediction averages over more examples. It may be more stable, but can blur important local patterns.

There is no universally best `k`. A common approach is to compare several values using cross-validation on training data, then evaluate the chosen model once on a separate test set.

For binary classification, an odd `k` can reduce tied votes, but it does not guarantee a good model.

## ⚖️ Uniform and Distance-Weighted Neighbors

KNN can give every selected neighbor equal influence, called **uniform weighting**.

It can also give closer neighbors more influence, called **distance weighting**. This can help when the closest examples should matter more than examples near the edge of the neighborhood.

Scikit-learn’s KNN classifiers and regressors support uniform and distance-based weighting options. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))

## 📐 Feature Scaling

KNN relies on distances, so features with large numerical ranges can dominate the calculation.

For example, suppose one feature is age, ranging roughly from 0 to 100, and another is annual income, ranging from 0 to 200,000. Without scaling, income may overwhelm age in a distance calculation even when both features are important.

Common scaling methods include:

- **Standardization:** Centers values around zero and scales them by their standard deviation.
- **Min-max scaling:** Maps values into a specified range, often 0 to 1.

Fit the scaler on training data only. Putting scaling and KNN in a scikit-learn pipeline helps prevent information leakage.

## 🧪 Python Example: Classification

This example uses scikit-learn’s built-in Iris dataset to classify flowers. It demonstrates the API; it is not intended for any high-stakes use.

Install scikit-learn:

```bash
pip install scikit-learn
```

Train and evaluate a classifier:

```python
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2,
    ),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
    )
)
```

The pipeline scales features using the training data and applies the same transformation when making predictions.

## 🧪 Python Example: Regression

This example uses a synthetic dataset to demonstrate KNN regression.

```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_regression(
    n_samples=1000,
    n_features=5,
    noise=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(
        n_neighbors=7,
        weights="distance",
    ),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

The data here is generated for demonstration. A real project should use representative data and an evaluation design suited to the problem.

## ⚙️ Important KNN Settings

Common scikit-learn settings include:

- `n_neighbors`: Number of neighbors to use.
- `weights`: How neighbors contribute; commonly `"uniform"` or `"distance"`.
- `metric`: Distance measure.
- `p`: Minkowski distance power; `1` is Manhattan and `2` is Euclidean.
- `algorithm`: Neighbor-search method, such as `"auto"`, `"ball_tree"`, `"kd_tree"`, or `"brute"`.
- `leaf_size`: Affects tree-based neighbor-search speed and memory use.

The best settings depend on the dataset. Scikit-learn documents these options for its KNN classifier and regressor. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))

## ⚠️ Strengths and Limitations

### Strengths

- Straightforward to understand and implement.
- Can model nonlinear decision boundaries.
- Supports classification and regression.
- Has few assumptions about the shape of the relationship.
- Can be a useful baseline for smaller datasets.

### Limitations

- Predictions can be slow when the training dataset is large, because the model may need to search many stored examples.
- It can require substantial memory because it retains the training data.
- Results can be sensitive to feature scaling and distance choice.
- Irrelevant or redundant features can distort distances.
- Performance can suffer in high-dimensional data, where distance becomes less informative.
- Class imbalance can dominate a majority vote.
- Predictions depend on having suitable nearby examples in the training data.

Scikit-learn describes nearest-neighbor methods as “non-generalizing” because they retain the training data, possibly in a transformed structure, and use it when answering queries. ([scikit-learn.org](https://scikit-learn.org/stable/modules/neighbors.html))

## 🧭 When to Use KNN

KNN is worth trying when:

- You want an understandable baseline.
- The dataset is small or moderate in size.
- Similar examples are likely to have similar outcomes.
- You can choose a meaningful distance measure.
- You can scale and prepare the features appropriately.

Consider another method when the dataset is very large, memory or prediction speed is limited, the feature space is high-dimensional, or distances do not represent meaningful similarity.

## 📝 Field Notes

- KNN predicts using nearby training examples.
- Classification uses a vote; regression commonly averages neighbor values.
- The value of `k`, the distance measure, and feature scaling all matter.
- Distance weighting lets closer examples influence predictions more strongly.
- KNN is simple, but prediction can become expensive as the dataset grows.
- Choose `k` through validation and evaluate on held-out data.

## 🔗 Further Reading

- [scikit-learn: Nearest Neighbors User Guide](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [scikit-learn: KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)

---

*KNN makes predictions by leaning on the examples nearest to a new case. Its simplicity is useful—but only when “near” is meaningful for the data.*
