
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from joblib import Parallel, delayed
import time

import random

def func(x):
    time.sleep(random.randint(1, 10))
    return x

def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time()-tick
        avg_speed = time_diff/step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)

all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}

def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun

aprun = ParallelExecutor(n_jobs=5)

a1 = aprun(total=25)(delayed(func)(i ** 2 + j) for i in range(5) for j in range(5))
a2 = aprun(total=16)(delayed(func)(i ** 2 + j) for i in range(4) for j in range(4))
a2 = aprun(bar='txt')(delayed(func)(i ** 2 + j) for i in range(4) for j in range(4))
a2 = aprun(bar=None)(delayed(func)(i ** 2 + j) for i in range(4) for j in range(4))

