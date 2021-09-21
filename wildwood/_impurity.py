# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains functions allowing to compute several impurity criteria in
nodes and their child nodes, and the information gain associated to it.
"""

from numba import jit, float32, uint32
from numba.types import Tuple
import numpy as np


@jit(float32(float32, float32, float32, float32), nopython=True, nogil=True)
def information_gain_proxy(
    impurity_left, impurity_right, w_samples_left, w_samples_right,
):
    """Computes a proxy of the information gain (improvement in impurity) using the
    equation

        - n_v0 * impurity_v0 - n_v1 * impurity_v1

    where:
        * n_v0, n_v1 are the weighted number of samples in the left and right nodes
        * impurity_v0, impurity_v1 are the impurities of the left and right nodes

    It is used in order to find the best split faster, by removing constant terms from
    the formula used in the information_gain function.

    Parameters
    ----------
    impurity_left : float
        Impurity of the left node

    impurity_right : float
        Impurity of the right node

    w_samples_left : float
        Weighted number of samples in the left node

    w_samples_right : float
        Weighted number of samples in the roght node

    Returns
    -------
    output : float
        Proxy of the information gain after splitting the parent into left
        and child nodes
    """
    return -w_samples_left * impurity_left - w_samples_right * impurity_right


@jit(
    float32(float32, float32, float32, float32, float32, float32, float32),
    nopython=True,
    nogil=True,
)
def information_gain(
    w_samples,
    w_samples_parent,
    w_samples_left,
    w_samples_right,
    impurity_parent,
    impurity_left,
    impurity_right,
):
    """Computes the information gain (improvement in impurity) using the equation

        n_v / n * impurity_v - n_v0 / n_v * impurity_v0 - n_v1 / n_v * impurity_v1)

    where:
        * v, v0, v1 are respectively the parent, left child and right child nodes
        * n is the total weighted number of samples
        * n_w and impurity_w are respectively the weighted number of samples and
          impurity of a node w

    Parameters
    ----------
    w_samples : float
        Total weighted number of samples

    w_samples_parent : float
        Weighted number of samples in parent node

    w_samples_left : float
        Weighted number of samples in left child node

    w_samples_right : float
        Weighted number of samples in right child node

    impurity_parent : float
        Impurity of the parent node

    impurity_left : float
        Impurity of the left child node

    impurity_right : float
        Impurity of the right child node

    Returns
    -------
    output : float
        Information gain obtained after splitting the parent into left and and
        right child nodes
    """
    return (w_samples_parent / w_samples) * (
        impurity_parent
        - (w_samples_right / w_samples_parent * impurity_right)
        - (w_samples_left / w_samples_parent * impurity_left)
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Gini impurity                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@jit(
    float32(uint32, float32, float32[::1]),
    nopython=True,
    nogil=True,
    locals={"y_sum_sq": float32, "w_samples_sq": float32},
)
def gini_node(n_classes, w_samples, y_sum):
    """Computes the gini impurity criterion in a node.

    Parameters
    ----------
    n_classes : int
        Number of label classes

    w_samples : float
        Weighted number of samples in the node

    y_sum : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class

    Returns
    -------
    output : float
        Gini impurity criterion in the node
    """
    y_sum_sq = 0.0
    w_samples_sq = w_samples * w_samples
    for k in range(n_classes):
        y_sum_sq += y_sum[k] * y_sum[k]
    return 1.0 - y_sum_sq / w_samples_sq


@jit(
    Tuple((float32, float32))(float32, float32, float32, float32[::1], float32[::1]),
    nopython=True,
    nogil=True,
    locals={
        "y_sum_left_sq": float32,
        "y_sum_right_sq": float32,
        "w_samples_left_sq": float32,
        "w_samples_right_sq": float32,
        "gini_left": float32,
        "gini_right": float32,
    },
)
def gini_childs(n_classes, w_samples_left, w_samples_right, y_sum_left, y_sum_right):
    """Computes the gini impurity criterion in both left and right child nodes of a
    parent node.

    Parameters
    ----------
    n_classes : int  # TODO: float32 in signature??
        Number of label classes

    w_samples_left : float
        Weighted number of samples in the left child node

    w_samples_right : float
        Weighted number of samples in the right child node

    y_sum_left : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class in the left child node

    y_sum_right : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class in the right child node

    Returns
    -------
    output : tuple
        A tuple of two floats containing the Gini impurities of the left child and
        right child nodes.
    """
    y_sum_left_sq = 0.0
    y_sum_right_sq = 0.0
    w_samples_left_sq = w_samples_left * w_samples_left
    w_samples_right_sq = w_samples_right * w_samples_right
    for k in range(n_classes):
        y_sum_left_sq += y_sum_left[k] * y_sum_left[k]
        y_sum_right_sq += y_sum_right[k] * y_sum_right[k]

    gini_left = 1.0 - y_sum_left_sq / w_samples_left_sq
    gini_right = 1.0 - y_sum_right_sq / w_samples_right_sq
    return gini_left, gini_right


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Entropy impurity                                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@jit(
    float32(uint32, float32, float32[::1]),
    nopython=True,
    nogil=True,
    locals={"entropy": float32, "y_sum_k": float32},
)
def entropy_node(n_classes, w_samples, y_sum):
    """Computes the entropy impurity criterion in a node.

    Parameters
    ----------
    n_classes : int
        Number of label classes

    w_samples : float
        Weighted number of samples in the node

    y_sum : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class

    Returns
    -------
    output : float
        Entropy impurity criterion in the node
    """
    entropy = 0.0
    for k in range(n_classes):
        y_sum_k = y_sum[k]
        if y_sum_k > 0.0:
            y_sum_k /= w_samples
            entropy -= y_sum_k * np.log(y_sum_k)
    return entropy


@jit(
    Tuple((float32, float32))(float32, float32, float32, float32[::1], float32[::1]),
    nopython=True,
    nogil=True,
    locals={
        "entropy_left": float32,
        "entropy_right": float32,
        "y_sum_left_k": float32,
        "y_sum_right_k": float32,
    },
)
def entropy_childs(n_classes, w_samples_left, w_samples_right, y_sum_left, y_sum_right):
    """Computes the entropy impurity criterion in both left and right child nodes of a
    parent node.

    Parameters
    ----------
    n_classes : int
        Number of label classes

    w_samples_left : float
        Weighted number of samples in the left child node

    w_samples_right : float
        Weighted number of samples in the right child node

    y_sum_left : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class in the left child node

    y_sum_right : ndarray
        Array of shape (n_classes,) and float dtype containing the weighted number of
        samples in each label class in the right child node

    Returns
    -------
    output : tuple
        A tuple of two floats containing the Entropy impurities of the left child and
        right child nodes.
    """
    entropy_left = 0.0
    entropy_right = 0.0
    for k in range(n_classes):
        y_sum_left_k = y_sum_left[k]
        y_sum_right_k = y_sum_right[k]
        if y_sum_left_k > 0.0:
            y_sum_left_k /= w_samples_left
            entropy_left -= y_sum_left_k * np.log(y_sum_left_k)
        if y_sum_right_k > 0.0:
            y_sum_right_k /= w_samples_right
            entropy_right -= y_sum_right_k * np.log(y_sum_right_k)

    return entropy_left, entropy_right


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Mean-squared error impurity                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@jit(float32(float32, float32, float32), nopython=True, nogil=True)
def mse_node(w_samples, y_sum, y_sq_sum):
    """Computes the variance of the labels in the node

    Parameters
    ----------
    w_samples : float
        Weighted number of samples in the node

    y_sum : float
        Weighted sum of the label in the node

    y_sq_sum : float
        Weighted sum of the squared labels in the node

    Returns
    -------
    output : float
        Variance of the labels in the node
    """
    return y_sq_sum / w_samples - (y_sum / w_samples) ** 2


@jit(
    Tuple((float32, float32))(float32, float32, float32, float32, float32, float32),
    nopython=True,
    nogil=True,
)
def mse_childs(
    w_samples_left,
    w_samples_right,
    y_sum_left,
    y_sum_right,
    y_sq_sum_left,
    y_sq_sum_right,
):
    """Computes the variance of the labels in both the left and right nodes of a
    parent node

    Parameters
    ----------
    w_samples_left : float
        Weighted number of samples in the left child node

    w_samples_right : float
        Weighted number of samples in the right child node

    y_sum_left : float
        Weighted sum of the label in the left child node

    y_sum_right : float
        Weighted sum of the label in the right child node

    y_sq_sum_left : float
        Weighted sum of the squared labels in the left child node

    y_sq_sum_right : float
        Weighted sum of the squared labels in the right child node

    Returns
    -------
    output : tuple
        A tuple with two float elements containing the variances of the label in
        the left and right child nodes
    """
    return (
        mse_node(w_samples_left, y_sum_left, y_sq_sum_left),
        mse_node(w_samples_right, y_sum_right, y_sq_sum_right),
    )
