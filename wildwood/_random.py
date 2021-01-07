# Authors: Arnaud Joly
#
# License: BSD 3 clause


import numpy as np
from ._utils import NP_UINT32_t, UINT32_t, njit

# cimport numpy as np
# ctypedef np.npy_uint32 UINT32_t
#
# cdef inline UINT32_t

DEFAULT_SEED = UINT32_t(1)

# cdef enum:
#     # Max value for our rand_r replacement (near the bottom).
#     # We don't use RAND_MAX because it's different across platforms and
#     # particularly tiny on Windows/MSVC.
#     RAND_R_MAX = 0x7FFFFFFF

RAND_R_MAX = UINT32_t(0x7FFFFFFF)


# cpdef sample_without_replacement(np.int_t n_population,
#                                  np.int_t n_samples,
#                                  method=*,
#                                  random_state=*)

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details

@njit
def our_rand_r(seed):
    # cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    # """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # # seed shouldn't ever be 0.
    # if (seed[0] == 0): seed[0] = DEFAULT_SEED
    #
    # seed[0] ^= <UINT32_t>(seed[0] << 13)
    # seed[0] ^= <UINT32_t>(seed[0] >> 17)
    # seed[0] ^= <UINT32_t>(seed[0] << 5)
    #
    # # Note: we must be careful with the final line cast to np.uint32 so that
    # # the function behaves consistently across platforms.
    # #
    # # The following cast might yield different results on different platforms:
    # # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    # #
    # # We can use:
    # # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # # or:
    # # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    # return seed[0] % <UINT32_t>(RAND_R_MAX + 1)
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if seed == 0:
        seed = DEFAULT_SEED

    seed ^= UINT32_t(seed << 13)
    seed ^= UINT32_t(seed >> 17)
    seed ^= UINT32_t(seed << 5)

    return seed % UINT32_t(RAND_R_MAX + 1), seed


@njit
def rand_int(low, high, random_state):
    r, seed = our_rand_r(random_state)
    return low + r % (high - low), seed


# cdef inline double rand_uniform(double low, double high,
#                                 UINT32_t* random_state) nogil:
#     """Generate a random double in [low; high)."""
#     return ((high - low) * <double> our_rand_r(random_state) /
#             <double> RAND_R_MAX) + low


@njit
def main():
    seed = 42
    for i in range(10):
        r, seed = rand_int(0, 10, seed)
        print(r)

# main()
