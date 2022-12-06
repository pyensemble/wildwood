import sys
from datetime import datetime
import logging
import numpy as np

import pandas as pd
import pickle
import argparse

import subprocess

sys.path.extend([".", ".."])
from wildwood.forest import ForestRegressor  # noqa: E402
from wildwood.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state_seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--n_snr_values", type=int, default=20)

    args = parser.parse_args()

    random_state_seed = args.random_state_seed
    n_repeats = args.n_repeats
    n_samples = args.n_samples
    n_snr_values = args.n_snr_values

    col_seed, col_mse, col_mae, col_snr, col_reg, col_signal = [], [], [], [], [], []

    random_seeds = list(range(random_state_seed, random_state_seed + n_repeats))

    logging.info(
        "Lauching regression experiment with hyperparameters n_samples = %d, n_repeats = %d, n_snr_values = %d"
        % (n_samples, n_repeats, n_snr_values)
    )

    signals = ["bumps", "heavisine", "blocks", "doppler"]
    signal_moments = [
        (0.05790276052474894, 0.022023680393615973),
        (0.5173006992227297, 0.3568104744362383),
        (0.4916027777777777, 0.312073225308642),
        (0.5474215781728211, 0.38576825810036947),
    ]
    regressors = {
        "RandomForestRegressor": RandomForestRegressor,
        "WildWood": ForestRegressor,
    }

    for i, seed in enumerate(random_seeds):
        logging.info("Repeat %d / %d" % (i+1, n_repeats))
        for ind in range(len(signals)):
            #logging.info("signal : %s" % signals[ind])
            for snr in np.logspace(0, 2, n_snr_values):
                noise_sigma = np.sqrt(signal_moments[ind][1] / snr)
                X, Y = make_regression(
                    n_samples=n_samples, random_state=seed, noise=noise_sigma, signal=signals[ind]
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=0.2, random_state=seed
                )

                for Reg in regressors.keys():

                    reg = regressors[Reg](n_estimators=100, random_state=seed)
                    reg.fit(X_train, y_train)

                    col_mse.append(mean_squared_error(y_test, reg.predict(X_test)))
                    col_mae.append(mean_absolute_error(y_test, reg.predict(X_test)))
                    col_seed.append(seed)
                    col_snr.append(snr)
                    col_reg.append(Reg)
                    col_signal.append(signals[ind])

    results = pd.DataFrame(
        {
            "seed": col_seed,
            "reg": col_reg,
            "mse": col_mse,
            "mae": col_mae,
            "snr": col_snr,
            "signal": col_signal,
        }
    )

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "exp_regression_" + now + ".pickle"

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "datetime": now,
                "commit": commit,
                "results": results,
            },
            f,
        )

    logging.info("Saved results in file %s" % filename)
