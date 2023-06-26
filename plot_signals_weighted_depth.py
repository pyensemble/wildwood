
import logging
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from wildwood.datasets import get_signal, make_regression
from wildwood.forest import ForestRegressor

from wildwood._binning import Binner

pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 2)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

colormap = get_cmap("tab20")

n_samples_train = 5000
n_samples_test = 1000
random_state = 42


noise = 0.03
aggregation = True
n_estimators = 100

step = 1 / noise ** 2

signal = "heavisine"

X_train, y_train = make_regression(
    n_samples=n_samples_train, signal=signal, noise=noise, random_state=random_state
)
X_test = np.linspace(0, 1, num=n_samples_test)

#
# reg = ForestRegressor(
#     random_state=random_state,
#     aggregation=aggregation,
#     max_features=1,
#     n_estimators=n_estimators,
#     step=step,
# )
#
# reg.fit(X_train.reshape(n_samples_train, 1), y_train)
# y_pred = reg.predict(X_test.reshape(n_samples_test, 1))
#
# df = reg.get_nodes(0)

# print(df)

# exit(0)

signals = ["heavisine", "bumps", "blocks", "doppler"]


def plot_weighted_depth(signal):

    X_train, y_train = make_regression(
        n_samples=n_samples_train, signal=signal, noise=noise, random_state=random_state
    )
    X_train = X_train.reshape(-1, 1)
    X_test = np.linspace(0, 1, num=n_samples_test).reshape(-1, 1)

    binner = Binner().fit(X_train)
    X_test_binned = binner.transform(X_test)

    reg = ForestRegressor(
        random_state=random_state,
        aggregation=aggregation,
        n_estimators=n_estimators,
        step=step,
    )

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    weighted_depths = reg._weighted_depth(X_test.reshape(n_samples_test, 1))

    # print("weighted_depths.shape:", weighted_depths.shape)

    # avg_weighted_depth = weighted_depths.mean(axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 5))

    plot_samples = ax1.plot(
        X_train, y_train, color=colormap.colors[1], lw=2, label="Samples"
    )[0]
    plot_signal = ax1.plot(
        X_test_binned / 255,
        get_signal(X_test_binned / 255, signal),
        lw=2,
        color=colormap.colors[0],
        label="Signal",
    )[0]
    plot_prediction = ax2.plot(
        X_test.ravel(), y_pred, lw=2, color=colormap.colors[2], label="Prediction"
    )[0]
    # ax3.plot(
    #     X_test,
    #     weighted_depths[:, 1:],
    #     lw=1,
    #     color=colormap.colors[5],
    #     alpha=0.2,
    #     label="Weighted depths",
    # )
    plot_weighted_depths = ax3.plot(
        X_test, weighted_depths.T, lw=1, color=colormap.colors[5], alpha=0.2
    )[0]

    plot_mean_weighted_depths = ax3.plot(
        X_test,
        weighted_depths.mean(axis=0),
        lw=2,
        color=colormap.colors[4],
        label="Mean weighted depth",
    )[0]
    filename = "weighted_depths_%s.pdf" % signal
    fig.subplots_adjust(hspace=0.1)
    fig.legend(
        (
            plot_signal,
            plot_samples,
            plot_mean_weighted_depths,
            plot_weighted_depths,
            plot_prediction,
        ),
        (
            "Signal",
            "Samples",
            "Average weighted depths",
            "Weighted depths",
            "Prediction",
        ),
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    # plt.savefig(filename)
    logging.info("Saved the decision functions in '%s'" % filename)


for signal in signals:
    plot_weighted_depth(signal)

plt.show()
