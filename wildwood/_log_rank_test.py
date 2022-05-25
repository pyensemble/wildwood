from numba import jit, float32, int32
import numpy as np
import pandas as pd
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH

@jit()
def survival_table_from_events(death_times, event_observed, birth_times, columns, weights):
    # deal with deaths and censors
    removed, observed, censored, entrance, at_risk = columns
    df = pd.DataFrame(death_times, columns=["event_at"])
    df[removed] = weights
    df[observed] = weights * ((event_observed).astype(bool))
    death_table = df.groupby("event_at").sum()
    death_table[censored] = (death_table[removed] - death_table[observed]).astype(int)

    # deal with late births
    births = pd.DataFrame(birth_times, columns=["event_at"])
    births[entrance] = weights
    births_table = births.groupby("event_at").sum()
    event_table = death_table.join(births_table, how="outer", sort=True).fillna(0)
    event_table["at_risk"] = event_table[entrance].cumsum() - event_table[
        removed].cumsum().shift(1).fillna(0)

    return event_table

@jit(
    float32(float32[::1], float32[::1], int32[::1], int32[::1]),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
)
def compute_test_statistic(event_times_1, event_times_2, event_observed_1, event_observed_2):
    event_times = np.hstack((event_times_1, event_times_2))
    event_observed = np.hstack((event_observed_1, event_observed_2))
    times =  np.unique(event_times[event_observed.astype(np.bool_)])
    O1_sum = event_observed_1.sum()
    O2_sum = event_observed_2.sum()
    E1_sum = 0.0
    E2_sum = 0.0
    for i, time in enumerate(times):
        # number at risk in group 1
        N1 = event_times_1[event_times_1 >= time].shape[0]
        # number at risk in group 2
        N2 = event_times_2[event_times_2 >= time].shape[0]
        if i == 0:
            # number at events in group 1
            O1 = event_observed_1[event_times_1 <= time].sum()
            # number at events in group 2
            O2 = event_observed_2[event_times_2 <= time].sum()
        else:
            # number at events in group 1
            O1 = event_observed_1[(event_times_1 <= time) & (event_times_1 > times[i-1])].sum()
            # number at events in group 2
            O2 = event_observed_2[(event_times_2 <= time) & (event_times_2 > times[i-1])].sum()
        E1_sum += N1 * (O1 + O2) / (N1 + N2)
        E2_sum += N2 * (O1 + O2) / (N1 + N2)

    test_statistic = (O1_sum - E1_sum)**2 / E1_sum + (O2_sum - E2_sum)**2 / E2_sum

    return test_statistic