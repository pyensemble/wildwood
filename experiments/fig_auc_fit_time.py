import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'results_all_2021-05-21-11-21-42.pickle'
# change to your pickle name which includes the concatenated dataframe

with open(filename, "rb") as f:
    results = pkl.load(f)

df = results["results"]


df_pivot = (
    df[["dataset", "classifier", "roc_auc", "fit_time"]]
    .pivot(index="dataset", columns="classifier")
)


datasets = [
    'adult',
    'breastcancer',
    'car',
    'covtype',
    'letter',
    'satimage',
    'sensorless',
    'spambase'
]


df_roc_auc = (
    df_pivot["roc_auc"]
    .reset_index()
    [["dataset", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "WildWood"]]
    .rename(
        columns={
            "XGBClassifier": "XGBoost",
            "LGBMClassifier": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "WildWood": "WildWood",
        }
    )
    .melt(id_vars=["dataset"])
)
df_roc_auc = df_roc_auc[df_roc_auc["dataset"].isin(datasets)]

df_fit_time = (
    df_pivot["fit_time"]
    .reset_index()
    [["dataset", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "WildWood"]]
    .rename(
        columns={
            "XGBClassifier": "XGBoost",
            "LGBMClassifier": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "WildWood": "WildWood",
        }
    )
    .melt(id_vars=["dataset"])
)
df_fit_time = df_fit_time[df_fit_time["dataset"].isin(datasets)]

sns.set_context("paper", font_scale=1.5)

f, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 5))
sns.barplot(x="dataset", y="value", hue="classifier", data=df_roc_auc, ax=ax1)
# ax.set_yscale("log")
# ax2.legend([])
# ax1.legend(ncol=7, loc="upper center", fontsize=17, bbox_to_anchor=(0.5, 1.3))
ax1.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.3))
ax1.set_xlabel(None)
# ax1.set_ylabel("AUC", fontsize=13)
ax1.set_ylabel("AUC")
# ax2.set_yticklabels(fontsize=15)
ax1.set_ylim((0.75, 1.04))
# g1.set_ticks(fontsize=14)
# ax1.set_xticklabels(datasets, fontsize=14)
# ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=12)
ax1.set_xticklabels([])
# ax1.set_yticks()
# plt.yticks(fontsize=12)

sns.barplot(x="dataset", y="value", hue="classifier", data=df_fit_time, ax=ax2)
ax2.set_xlabel(None)
ax2.set_yscale("log")
# ax2.legend(ncol=7, loc="upper left", fontsize=15)
# ax2.legend([], )
ax2.get_legend().remove()
# ax2.set_ylabel("Fit time (seconds)", fontsize=13)
ax2.set_ylabel("time (sec.)")
# plt.xticks(fontsize=14)
# ax2.set_xticklabels(datasets, fontsize=15)
ax2.set_xticklabels(datasets)
# plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("fig_auc_timings.pdf")
