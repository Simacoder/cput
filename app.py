import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")


# ---------- Data & model (cached so they only run once) ----------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, df


@st.cache_resource
def train_model(n_estimators: int, test_size: float, seed: int):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return {
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, target_names=iris.target_names, output_dict=True
        ),
        "cm": confusion_matrix(y_test, y_pred),
    }


iris, df = load_data()
feature_names = iris.feature_names
class_names = list(iris.target_names)

# ---------- Sidebar ----------
st.sidebar.header("Model settings")
n_estimators = st.sidebar.slider("Number of trees", 10, 300, 100, step=10)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

res = train_model(n_estimators, test_size, int(seed))

st.title("🌸 Iris Flower Classifier")
st.caption("Random Forest trained on the classic Iris dataset.")

tab_predict, tab_eda, tab_eval = st.tabs(["Predict", "Explore data", "Model performance"])

# ---------- Predict tab ----------
with tab_predict:
    st.subheader("Enter flower measurements (cm)")
    col_in, col_out = st.columns([1, 1])

    with col_in:
        inputs = []
        for name in feature_names:
            lo, hi = float(df[name].min()), float(df[name].max())
            val = st.slider(
                name.title(), min_value=round(lo - 0.5, 1), max_value=round(hi + 0.5, 1),
                value=round(float(df[name].mean()), 1), step=0.1,
            )
            inputs.append(val)

    with col_out:
        x_new = res["scaler"].transform(np.array(inputs).reshape(1, -1))
        pred = res["model"].predict(x_new)[0]
        proba = res["model"].predict_proba(x_new)[0]

        st.metric("Predicted species", class_names[pred].title())
        st.write("Class probabilities")
        proba_df = pd.DataFrame({"species": class_names, "probability": proba}).set_index("species")
        st.bar_chart(proba_df)

        st.write("Feature importance")
        imp = pd.Series(res["model"].feature_importances_, index=feature_names).sort_values()
        st.bar_chart(imp)

# ---------- EDA tab ----------
with tab_eda:
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True, height=250)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Summary statistics")
        st.dataframe(df.describe())
    with c2:
        st.write("Class distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="species", data=df, ax=ax)
        st.pyplot(fig)

    st.write("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("Scatter plot")
    x_axis = st.selectbox("X axis", feature_names, index=2)
    y_axis = st.selectbox("Y axis", feature_names, index=3)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="species", ax=ax)
    st.pyplot(fig)

    if st.checkbox("Show pairplot (slower)"):
        st.pyplot(sns.pairplot(df, hue="species"))

# ---------- Evaluation tab ----------
with tab_eval:
    st.subheader("Test-set results")
    st.metric("Accuracy", f"{res['accuracy']:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Classification report")
        st.dataframe(pd.DataFrame(res["report"]).T.round(2))
    with c2:
        st.write("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            res["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
