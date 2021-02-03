"""
This module contains functions allowing to compute impurity criterions in nodes and
their childs, and the information gain associated to it
"""

# TODO: rename this module _node and put in there all the impurities and
#  loss functions

from ._utils import njit, nb_float32, nb_uint32


# @njit(float32(float32, float32, float32, float32))

@njit
def information_gain_proxy(
    impurity_left,
    impurity_right,
    n_samples_left,
    n_samples_right,
):
    """Computes a proxy of the information gain (improvement in impurity) using the
    equation

        - n_v0 * impurity_v0 - n_v1 * impurity_v1

    where:
        * v0, v1 are respectively the left child and right child nodes
        * n_w and impurity_w are respectively the weighted number of samples and
          impurity of a node w

    It is used in order to find the best split faster, by removing constant terms from
    the formula used in the information_gain function.

    Parameters
    ----------
    get_childs_impurity : callable
        A function computing the impurity of the childs


    Return
    ------
    output : float32
        Information gain after splitting parent into left and child nodes
    """
    # impurity_left, impurity_right = get_childs_impurity(
    #     n_classes, n_samples_left, n_samples_right, y_sum_left, y_sum_right
    # )
    return - n_samples_left * impurity_left - n_samples_right * impurity_right


# @njit(float32(float32, float32, float32, float32, float32, float32, float32))

@njit
def information_gain(
    n_samples,
    n_samples_parent,
    n_samples_left,
    n_samples_right,
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
    n_samples : float32
        Total weighted number of samples
    n_samples_parent : float32
        Weighted number of samples in parent node
    n_samples_left : float32
        Weighted number of samples in left child node
    n_samples_right : float32
        Weighted number of samples in right child node
    impurity_parent : float32
        Impurity of the parent node
    impurity_left : float32
        Impurity of the left child node
    impurity_right : float32
        Impurity of the right child node

    Return
    ------
    output : float32
        Information gain after splitting parent into left and child nodes
    """
    return (n_samples_parent / n_samples) * (
        impurity_parent
        - (n_samples_right / n_samples_parent * impurity_right)
        - (n_samples_left / n_samples_parent * impurity_left)
    )


# (nb_float32(nb_uint32, nb_float32, nb_float32[::1]))
@njit
def gini_node(n_classes, n_samples, y_sum):
    """Computes the gini impurity criterion in a node

    Parameters
    ----------
    n_classes : uint
        Number of label classes
    n_samples : float32
        Number of samples in the node (weighted number of samples if the
        sample_weights is used)
    y_sum : array of float32 with shape (n_classes,)
        Array containing the number of samples for each class (weighted count if the
        sample_weights is used)
    """
    y_sum_sq = 0.0
    n_samples_sq = n_samples * n_samples
    for k in range(n_classes):
        y_sum_sq += y_sum[k] * y_sum[k]
    return 1.0 - y_sum_sq / n_samples_sq


# TODO: spec with Tuple return type ?
@njit
def gini_childs(n_classes, n_samples_left, n_samples_right, y_sum_left, y_sum_right):
    # TODO: docstring
    y_sum_left_sq = 0.0
    y_sum_right_sq = 0.0
    # print("n_samples_left: ", n_samples_left, ", n_samples_right: ", n_samples_right)
    n_samples_left_sq = n_samples_left * n_samples_left
    n_samples_right_sq = n_samples_right * n_samples_right
    for k in range(n_classes):
        y_sum_left_sq += y_sum_left[k] * y_sum_left[k]
        y_sum_right_sq += y_sum_right[k] * y_sum_right[k]

    gini_left = 1.0 - y_sum_left_sq / n_samples_left_sq
    gini_right = 1.0 - y_sum_right_sq / n_samples_right_sq
    return gini_left, gini_right

