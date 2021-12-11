from time import time
import numpy as np

from wildwood.preprocessing.features_bitarray import (
    features_bitarray_to_array,
    array_to_bitarray,
)


np.random.seed(42)


def simulate_X(n_samples, max_values):
    n_features = max_values.size
    return np.asfortranarray(
        # high is not incuded in randint hence the + 1
        np.random.randint(max_values + 1, size=(n_samples, n_features))
    )


def test_dataset():
    n_samples = 10_000
    # n_samples = 1
    max_values = np.array([17, 3, 69], dtype=np.uint64)
    # max_values = np.array([2378, 16, 123, 213, 123765, 7], dtype=np.uint64)
    # max_values = np.array([2378, 2, 16, 123, 1, 213, 123765, 7, 8, 32, 64, 1024, 3],
    #                       dtype=np.uint64)
    # max_values = np.array([1], dtype=np.uint64)
    # max_values = np.array([2], dtype=np.uint64)
    n_features = max_values.size
    X_in = np.asfortranarray(
        np.random.randint(max_values + 1, size=(n_samples, n_features)), dtype=np.uint64
    )

    # X_in = simulate_X(n_samples, max_values)

    # dataset = Dataset(n_samples, max_values)

    dataset = array_to_bitarray(X_in, max_values=max_values)

    print("n_values_in_words:", dataset.n_values_in_words)
    print("n_bits:", dataset.n_bits)
    print("offsets:", dataset.offsets)
    print("bitmasks:", dataset.bitmasks)

    # print("bitarray:", dataset.bitarray)

    # dataset_fill_values(dataset, X)

    X_out = features_bitarray_to_array(dataset)

    # print(X_in - X_out)
    # print(X_in)
    # print(X_out)

    np.testing.assert_array_equal(X_in, X_out)
    # Xc = _get_empty_matrix(dataset)
    # dataset_to_array(dataset, Xc)


test_dataset()


exit(0)


def test_correct():
    n_samples = 13
    max_value = 2
    values = np.random.randint(0, max_value, n_samples, dtype=np.uint64)
    # values = np.array([3, 42, 13, 57, 17, 8, 10, 11, 17, 19, 34, 24], dtype=uint64)
    # n_samples = values.size
    max_value_ = values.max()
    column = Column(n_samples, max_value_, values)

    print_column(column)

    column_set_values(column, values)

    print(values)
    for i in range(n_samples):
        print(column_get(column, i))


# @jit(nogil=True, nopython=True)
def test_speed():
    n_samples = 50_000_000
    values = np.random.randint(0, 56, n_samples, dtype=np.uint64)
    # values = np.array([3, 42, 13, 57, 17, 8, 10, 11, 17, 19, 34, 24], dtype=uint64)
    # n_samples = values.size
    max_value = values.max()
    column = Column(n_samples, max_value, values)
    print_column(column)

    column_set_values(column, values)
    idx = np.arange(column.n_samples)
    np.random.shuffle(idx)

    s = loop(column, idx)
    s = loop2(column, idx)

    tic = time()
    column_set_values(column, values)
    toc = time()
    print("column_set_values: %.2e" % (toc - tic))

    tic = time()
    s = loop(column, idx)
    toc = time()
    print("loop: %.2e   s: %d" % (toc - tic, s))

    tic = time()
    s = loop2(column, idx)
    toc = time()
    print("loop2: %.2e   s: %d" % (toc - tic, s))

    # for i in range(n_samples):
    #     print(column_get(column, i))


if __name__ == "__main__":
    bitset = test_correct()
