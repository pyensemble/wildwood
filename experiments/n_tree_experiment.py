# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This script produces Figure 2 from the WildWood's paper.
"""


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


import sys
import subprocess
from datetime import datetime
import numpy as np
import logging
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

sys.path.extend([".", ".."])

from wildwood.datasets import load_adult, load_bank, load_car, load_default_cb
from wildwood.forest import ForestClassifier
    
import warnings
warnings.filterwarnings('ignore')

loaders = [load_adult, load_bank, load_default_cb, load_car]

random_state = 42

classifiers = [
    lambda n: (
        "RFW",
        RandomForestClassifier(
            n_estimators=n,
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
    lambda n: (
        "WildWood",
        ForestClassifier(
            n_estimators=n,
            multiclass="multinomial",
            n_jobs=-1,
            random_state=random_state,
            handle_unknown="consider_missing",
        ),
    ),
    lambda n: (
        "ET",
        ExtraTreesClassifier(
            n_estimators=n,
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

data_extraction = {
    "RandomForestClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "WildWood": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "ForestClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "ExtraTreesClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
}


def fit_kwargs_generator(clf_name, y_train, dataset):
    if clf_name == "RandomForestClassifier":
        return {}
    elif clf_name == "ForestClassifier":
        return {"categorical_features": dataset.categorical_features_}
    elif clf_name == "ExtraTreesClassifier":
        return {}
    else:
        print("ERROR : NOT Found : ", clf_name)


data_random_states = list(range(42, 42 + 10))

col_data = []
col_classifier = []
col_classifier_title = []
col_n_trees = []
col_x_pos = []
col_repeat = []
col_roc_auc = []
col_roc_auc_weighted = []
col_avg_precision_score = []
col_avg_precision_score_weighted = []
col_log_loss = []
col_accuracy = []

n_datasets = None  # set to None to use all
n_treess = [1, 2, 5, 10, 20, 50, 100, 200]

for x, n in enumerate(n_treess):
    for Clf in classifiers:
        clf_title, clf = Clf(n)
        clf_name = clf.__class__.__name__
        logging.info("-" * 64)
        logging.info("classifier : %s , n_trees = %d" % (clf_name, n))
        for loader in loaders[:n_datasets]:
            dataset = loader()
            data_name = dataset.name
            task = dataset.task
            for key, val in data_extraction[clf_name].items():
                setattr(dataset, key, val)
            logging.info("-" * 64)
            logging.info("Dataset : %r" % data_name)
            for repeat, data_random_state in enumerate(data_random_states):
                clf_title, clf = Clf(n)
                col_data.append(data_name)
                col_classifier.append(clf_name)
                col_classifier_title.append(clf_title)
                col_n_trees.append(n)
                col_x_pos.append(x + 1)
                col_repeat.append(repeat)

                X_train, X_test, y_train, y_test = dataset.extract(
                    random_state=data_random_state
                )
                y_test_binary = LabelBinarizer().fit_transform(y_test)

                clf.fit(
                    X_train,
                    y_train,
                    **(fit_kwargs_generator(clf_name, y_train, dataset))
                )

                y_scores = clf.predict_proba(X_test)
                y_pred = np.argmax(y_scores, axis=1)

                if task == "binary-classification":
                    roc_auc = roc_auc_score(y_test, y_scores[:, 1])
                    roc_auc_weighted = roc_auc
                    avg_precision_score = average_precision_score(
                        y_test, y_scores[:, 1]
                    )
                    avg_precision_score_weighted = avg_precision_score
                    log_loss_ = log_loss(y_test, y_scores)
                    accuracy = accuracy_score(y_test, y_pred)
                elif task == "multiclass-classification":
                    roc_auc = roc_auc_score(
                        y_test, y_scores, multi_class="ovr", average="macro"
                    )
                    roc_auc_weighted = roc_auc_score(
                        y_test, y_scores, multi_class="ovr", average="weighted"
                    )
                    avg_precision_score = average_precision_score(
                        y_test_binary, y_scores
                    )
                    avg_precision_score_weighted = average_precision_score(
                        y_test_binary, y_scores, average="weighted"
                    )
                    log_loss_ = log_loss(y_test, y_scores)
                    accuracy = accuracy_score(y_test, y_pred)
                else:
                    raise ValueError("Task %s not understood" % task)

                col_roc_auc.append(roc_auc)
                col_roc_auc_weighted.append(roc_auc_weighted)
                col_avg_precision_score.append(avg_precision_score)
                col_avg_precision_score_weighted.append(avg_precision_score_weighted)
                col_log_loss.append(log_loss_)
                col_accuracy.append(accuracy)


results = pd.DataFrame(
    {
        "dataset": col_data,
        "classifier": col_classifier,
        "classifier_title": col_classifier_title,
        "repeat": col_repeat,
        "n_trees": col_n_trees,
        "x_pos": col_x_pos,
        "roc_auc": col_roc_auc,
        "roc_auc_w": col_roc_auc_weighted,
        "avg_prec": col_avg_precision_score,
        "avg_prec_w": col_avg_precision_score_weighted,
        "log_loss": col_log_loss,
        "accuracy": col_accuracy,
    }
)

def plot_comparison_n_trees(df, metric="roc_auc", filename=None, legend=True):
    g = sns.FacetGrid(
        df, col="dataset", col_wrap=4, aspect=1, height=4, sharex=True, sharey=False
    )
    g.map(
        sns.lineplot,
        "x_pos",
        metric,
        "classifier",
        lw=4,
        marker="o",
        markersize=10,
    ).set(xlabel="", ylabel="")

    axes = g.axes.flatten()
    y_ticks = [0,1, 2, 5, 10, 20, 50, 100, 200]
    for i, dataset in enumerate(df["dataset"].unique()):        
        axes[i].xaxis.set_ticks(list(range(len(y_ticks))))
        axes[i].set_xticklabels(y_ticks, fontsize=14)
        left,right = axes[i].get_xlim()
        axes[i].set_xlim(0.6, right)
        axes[i].set_title(dataset, fontsize=20)
        axes[i].set_xlabel("#Trees", fontsize=18, labelpad=0.0)
        axes[i].tick_params(axis='y', which="major", labelsize=14)
        axes[i].tick_params(axis='y', which="minor", labelsize=14)
        #plt.yticks(fontsize=14)
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%.2g'))
        
    if legend:
        plt.legend(
            handles=axes[-1].lines,
            labels=["RandomForest", "WildWood", "ExtraTrees"],
            bbox_to_anchor=(0.0, 0.45, 1.0, 0.0),
            loc="upper right",
            ncol=1,
            borderaxespad=0.0,
            fontsize=18,
        )

    #plt.tight_layout()

    
plot_comparison_n_trees(results, metric="roc_auc")

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

plt.savefig("fig_ntrees_"+now+".pdf")

plt.show()

