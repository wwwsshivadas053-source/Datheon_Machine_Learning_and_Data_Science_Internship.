"""
Multi-Class Benchmark Classification Pipeline (Iris Dataset)
Task 1: End-to-end classification pipeline with Scikit-Learn, data normalization, 
confusion matrices, and decision boundary visualization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Set style for professional figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset(csv_path="iris.csv"):
    """Load the benchmark dataset using pandas."""
    print("=" * 70)
    print("STEP 1: LOADING & INSPECTING DATASET")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df["species"].value_counts())
    print("\nSummary Statistics:")
    print(df.describe())
    return df


def perform_eda(df):
    """Perform exploratory data analysis and save pairplots and correlations."""
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (PAIRPLOT & CORRELATION)")
    print("=" * 70)
    
    # 1. Pairplot
    pairplot_fig = sns.pairplot(
        df,
        hue="species",
        palette={"setosa": "#2b5c8f", "versicolor": "#d95f02", "virginica": "#7570b3"},
        markers=["o", "s", "D"],
        diag_kind="kde",
        corner=False
    )
    pairplot_fig.fig.suptitle("Iris Dataset Pairwise Feature Distributions (by Species)", y=1.02, fontsize=14, weight="bold")
    pairplot_path = os.path.join(OUTPUT_DIR, "eda_pairplot.png")
    pairplot_fig.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pairplot to: {pairplot_path}")

    # 2. Correlation Heatmap
    numeric_df = df.drop(columns=["species"])
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=1,
        linecolor="white",
    )
    plt.title("Iris Feature Correlation Matrix", fontsize=14, weight="bold", pad=15)
    corr_path = os.path.join(OUTPUT_DIR, "eda_correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to: {corr_path}")


def prepare_and_split_data(df, test_size=0.20, random_state=42):
    """
    Split data into 80/20 train/test sets with stratification
    and apply StandardScaler to prevent data leakage.
    """
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN / TEST SPLIT & FEATURE NORMALIZATION (StandardScaler)")
    print("=" * 70)
    
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples ({1-test_size:.0%})")
    print(f"Testing set:  {X_test.shape[0]} samples ({test_size:.0%})")
    print("Class breakdown in test set:")
    print(y_test.value_counts())

    # Fit scaler strictly on training set to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Summary (StandardScaler):")
    for i, col in enumerate(feature_cols):
        print(f"  {col:<15} Train Mean={scaler.mean_[i]:.3f}, Train Std={np.sqrt(scaler.var_[i]):.3f}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_models(X_train_scaled, y_train, random_state=42):
    """Train Logistic Regression and Decision Tree models."""
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING (LOGISTIC REGRESSION & DECISION TREE)")
    print("=" * 70)

    # 1. Logistic Regression
    lr_model = LogisticRegression(
        max_iter=500,
        random_state=random_state,
        solver="lbfgs",
        C=1.0
    )
    lr_model.fit(X_train_scaled, y_train)
    print("Logistic Regression trained successfully.")

    # 2. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=random_state
    )
    dt_model.fit(X_train_scaled, y_train)
    print("Decision Tree Classifier trained successfully (max_depth=3).")

    return lr_model, dt_model


def evaluate_and_benchmark(models_dict, X_test_scaled, y_test, labels):
    """Benchmark classification accuracy, precision, recall, and F1-score."""
    print("\n" + "=" * 70)
    print("STEP 5: BENCHMARK CLASSIFICATION METRICS & REPORTS")
    print("=" * 70)

    benchmark_records = []
    predictions_dict = {}

    for name, model in models_dict.items():
        y_pred = model.predict(X_test_scaled)
        predictions_dict[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        
        prec_weighted = precision_score(y_test, y_pred, average="weighted")
        rec_weighted = recall_score(y_test, y_pred, average="weighted")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        benchmark_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec_macro,
            "Recall (Macro)": rec_macro,
            "F1-Score (Macro)": f1_macro,
            "Precision (Weighted)": prec_weighted,
            "Recall (Weighted)": rec_weighted,
            "F1-Score (Weighted)": f1_weighted,
        })

        print(f"\n--- Classification Report: {name} ---")
        print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    benchmark_df = pd.DataFrame(benchmark_records)
    print("\nBENCHMARK COMPARISON TABLE:")
    print(benchmark_df.to_string(index=False))

    return benchmark_df, predictions_dict


def plot_confusion_matrices(y_test, predictions_dict, labels):
    """Plot side-by-side confusion matrix heatmaps using Seaborn."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=1.5,
            linecolor="white",
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Predicted Species", fontsize=11, weight="bold")
        ax.set_ylabel("True Species", fontsize=11, weight="bold")

    plt.suptitle("Model Evaluation: Confusion Matrix Comparison", fontsize=15, weight="bold", y=1.03)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrix comparison to: {cm_path}")


def plot_benchmark_chart(benchmark_df):
    """Plot comparison bar chart for all benchmark metrics."""
    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    melted_df = benchmark_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#1f77b4", "#ff7f0e"]
    )
    plt.title("Classification Benchmark: Logistic Regression vs Decision Tree", fontsize=13, weight="bold", pad=12)
    plt.ylabel("Score", fontsize=11, weight="bold")
    plt.xlabel("Evaluation Metric", fontsize=11, weight="bold")
    plt.ylim(0.8, 1.05)
    plt.legend(loc="lower right", frameon=True)

    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9, weight="bold",
                        xytext=(0, 3), textcoords='offset points')

    chart_path = os.path.join(OUTPUT_DIR, "benchmark_metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved benchmark metric comparison to: {chart_path}")


def plot_decision_boundaries(df, labels):
    """
    Train 2D models on the most informative features (petal length vs petal width)
    and plot clear decision boundaries with the data points.
    """
    print("\n" + "=" * 70)
    print("STEP 6: DECISION BOUNDARY VISUALIZATION (2D FEATURE SPACE)")
    print("=" * 70)

    # Label encoding for visual plotting
    label_mapping = {l: i for i, l in enumerate(labels)}
    reverse_mapping = {i: l for i, l in enumerate(labels)}

    X_2d = df[["petal_length", "petal_width"]].values
    y_num = df["species"].map(label_mapping).values

    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)

    # Train 2D models
    lr_2d = LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")
    lr_2d.fit(X_2d_scaled, y_num)

    dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_2d.fit(X_2d_scaled, y_num)

    # Create meshgrid
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.7, X_2d_scaled[:, 0].max() + 0.7
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.7, X_2d_scaled[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    models_2d = [("Logistic Regression (Linear Boundaries)", lr_2d),
                 ("Decision Tree (Orthogonal Axis-Aligned Boundaries)", dt_2d)]

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(["#c6dbef", "#fdd0a2", "#dadaeb"])
    cmap_bold = ListedColormap(colors)

    for ax, (title, model) in zip(axes, models_2d):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision region contour
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        ax.contour(xx, yy, Z, colors="black", linewidths=0.7, alpha=0.5)

        # Scatter actual data points
        scatter = ax.scatter(
            X_2d_scaled[:, 0],
            X_2d_scaled[:, 1],
            c=y_num,
            cmap=cmap_bold,
            edgecolor="k",
            s=45,
            alpha=0.9
        )

        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Petal Length (Standardized)", fontsize=11)
        ax.set_ylabel("Petal Width (Standardized)", fontsize=11)

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=l,
               markerfacecolor=c, markersize=9, markeredgecolor='k')
        for l, c in zip(labels, colors)
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=11)

    plt.suptitle("Decision Boundary Comparison: Petal Length vs Petal Width", fontsize=15, weight="bold", y=1.09)
    boundary_path = os.path.join(OUTPUT_DIR, "decision_boundaries.png")
    plt.tight_layout()
    plt.savefig(boundary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved decision boundaries plot to: {boundary_path}")


def plot_tree_diagram(dt_model, feature_names, labels):
    """Visualize the Decision Tree structure."""
    plt.figure(figsize=(14, 8))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=labels,
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2,
    )
    plt.title("Decision Tree Classifier (max_depth=3) Internal Split Architecture", fontsize=14, weight="bold", pad=12)
    tree_path = os.path.join(OUTPUT_DIR, "decision_tree_structure.png")
    plt.tight_layout()
    plt.savefig(tree_path, dpi=300)
    plt.close()
    print(f"Saved Decision Tree diagram to: {tree_path}")


def main():
    print("=" * 70)
    print("RUNNING BENCHMARK CLASSIFICATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset("iris.csv")
    labels = sorted(df["species"].unique().tolist())

    # 2. EDA
    perform_eda(df)

    # 3. Train/Test Split (80/20) & Normalization (StandardScaler)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = prepare_and_split_data(df)

    # 4. Train Models
    lr_model, dt_model = train_models(X_train_scaled, y_train)
    models_dict = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model
    }

    # 5. Evaluate and Benchmark
    benchmark_df, predictions_dict = evaluate_and_benchmark(models_dict, X_test_scaled, y_test, labels)

    # 6. Plot Confusion Matrices
    plot_confusion_matrices(y_test, predictions_dict, labels)

    # 7. Plot Benchmark Metrics Chart
    plot_benchmark_chart(benchmark_df)

    # 8. Plot Decision Boundaries
    plot_decision_boundaries(df, labels)

    # 9. Plot Decision Tree Diagram
    plot_tree_diagram(dt_model, feature_cols, labels)

    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
