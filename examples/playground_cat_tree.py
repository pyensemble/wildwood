import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import colorcet as cc
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)
from sklearn.metrics import roc_auc_score, log_loss

sys.path.extend([".", ".."])

from wildwood.forest import ForestBinaryClassifier


@st.cache
def get_mesh(X, h=0.02, padding=0.5):
    """Build a regular meshgrid using the range of the features in X
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.array([xx.ravel(), yy.ravel()]).T
    return xx, yy, X_mesh


def plot_scatter_binary_classif(
    ax,
    xx,
    yy,
    X,
    y,
    s=10,
    alpha=None,
    cm=None,
    title=None,
    fontsize=None,
    lw=None,
    norm=None,
    noaxes=False,
):
    if cm is None:
        cm = plt.get_cmap("RdBu")

    ax.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=cm, alpha=alpha, lw=lw, norm=norm)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if noaxes:
        ax.axis("off")


def plot_contour_binary_classif(
    ax, xx, yy, Z, cm=None, alpha=0.8, levels=200, title=None, score=None, norm=None
):
    if cm is None:
        cm = plt.get_cmap("RdBu")
    ax.contourf(xx, yy, Z, cmap=cm, alpha=alpha, levels=levels, norm=norm)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title)
    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )


@st.cache
def simulate_data(random_state=42):
    X = np.random.randint(0, 5, (10, 5))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y, X[:7], y[:7], X[7:], y[7:]


@st.cache
def compute_features_range(X_train, eps=0.5):
    x_min = X_train[:, 0].min() - eps
    x_max = X_train[:, 0].max() + eps
    y_min = X_train[:, 1].min() - eps
    y_max = X_train[:, 1].max() + eps
    return x_min, x_max, y_min, y_max


def fit_forest(
    X_train,
    y_train,
    aggregation=True,
    n_estimators=10,
    dirichlet=0.5,
    step=1.0,
    min_samples_split=2,
    n_jobs=1,
):

    clf_kwargs = {
        "n_estimators": n_estimators,
        "aggregation": aggregation,
        "min_samples_split": min_samples_split,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "step": step,
        "dirichlet": dirichlet,
    }
    clf = ForestBinaryClassifier(**clf_kwargs)
    clf.fit(X_train, y_train)
    return clf

def get_decision(clf, X_mesh):
    zz = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    return zz


@st.cache
def get_normalizer(normalize):
    if normalize:
        return plt.Normalize(vmin=0.0, vmax=1.0)
    else:
        return None


st.title("`WildWood` playground")
st.sidebar.title("Dataset")
st.sidebar.markdown("Choose the dataset below")
dataset = st.sidebar.selectbox(
    "dataset", ["moons", "circles", "linear", "blobs"], index=0
)
st.sidebar.title("Parameters")
st.sidebar.markdown(
    """You can tune below some 
hyperparameters"""
)
n_estimators = st.sidebar.selectbox("n_estimators", [1, 5, 10, 50, 100], index=2)
aggregation = st.sidebar.checkbox("aggregation", value=True)
dirichlet = st.sidebar.selectbox(
    "dirichlet", [1e-8, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 100.], index=3
)
step = st.sidebar.selectbox(
    "step", [1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0], index=3
)
show_data = st.sidebar.checkbox("Show data", value=True)
normalize = st.sidebar.checkbox("Normalize colors", value=True)


n_samples = 200
random_state = 42
grid_size = 200
levels = 20


# X, y, X_train, y_train, X_test, y_test = simulate_data(dataset, random_state=42)
X, y, X_train, y_train, X_test, y_test = simulate_data(random_state=42)
clf = fit_forest(
    X_train,
    y_train,
    aggregation,
    n_estimators,
    dirichlet,
    step,
    n_jobs=-1
)
xx, yy, X_mesh = get_mesh(X)
zz = get_decision(clf, X_mesh)
norm = get_normalizer(normalize)

y_pred = clf.predict_proba(X_test)[:, 1]


score_test = log_loss(y_test, y_pred)

# score_test = roc_auc_score(y_test, y_pred)

_ = plt.figure(figsize=(3, 3))
ax = plt.subplot(1, 1, 1)
plot_contour_binary_classif(ax, xx, yy, zz, levels=levels, norm=norm, score=score_test)
if show_data:
    plot_scatter_binary_classif(ax, xx, yy, X_train, y_train, s=5, lw=1, norm=norm)

plt.tight_layout()
st.pyplot()
