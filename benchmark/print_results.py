import pandas as pd
import mlflow


pd.set_option("display.width", 160)
pd.set_option("display.precision", 3)
EXPERIMENT_NAME = "ww_bench_v1"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id
runs = mlflow.search_runs(experiment_ids=experiment_id)


columns = [
    "dataset",
    "fit_time",
    "predict_train_time",
    "predict_test_time",
    "log_loss_train",
    "log_loss_test",
    "roc_auc_train",
    "roc_auc_test",
]

runs_avg = (
    runs[
        [
            col
            for col in runs.columns
            if col.startswith("metrics") or col.startswith("params")
        ]
    ]
    .rename(
        {
            key: key.replace("metrics.", "").replace("params.", "")
            for key in runs.keys()
        },
        axis="columns",
    )[columns]
    .groupby(["dataset"])
    .agg("mean")
)

print("=" * 64 + " DETAILS " + "=" * 64)
print(runs_avg)
print("=" * 64 + " OVERALL MEAN " + "=" * 64)
print(runs_avg.agg(["mean", "sum"]))
