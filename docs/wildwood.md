
(description-wildwood)=
# Description of WildWood

\begin{equation*}
\newcommand{\cX}{\mathcal X}
\newcommand{\cY}{\mathcal Y}
\newcommand{\bs}{\boldsymbol}
\newcommand{\cell}{C}
\newcommand{\node}{\mathbf{v}}
\newcommand{\leaves}{\mathrm{leaves}}
\newcommand{\leaf}{\node}%{\mathbf{l}}
\newcommand{\tree}{\mathcal{T}}
\newcommand{\splits}{\Sigma}
\newcommand{\pred}{\widehat{y}}
\newcommand{\otb}{{\mathtt{otb}}}
\newcommand{\itb}{{\mathtt{itb}}}
\newcommand{\probas}{\mathcal{P}}
\newcommand{\P}{\mathcal{P}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\dirichletdist}{\mathsf{Dir}}
\newcommand{\nodes}{\mathrm{nodes}}
\newcommand{\inodes}{\mathrm{intnodes}}
\renewcommand{\root}{\mathtt{root}}
\renewcommand{\P}{\mathbb P}
\newcommand{\ind}[1]{\mathbf 1_{#1}}
\newcommand{\loss}{\ell}
\newcommand{\pp}{\; : \; }
\newcommand{\eps}{\varepsilon}
\newcommand{\wh}{\widehat}
\newcommand{\pathpoint}{\mathtt{path}}
\newcommand{\wbar}{w^{\mathrm{den}}}
\newcommand{\wnum}{w^{\mathrm{num}}}
\end{equation*}

Let us explain here a bit of the theory behind `WildWood`.
We have data that comes as a set of training samples 
$(x_i, y_i)$ for $i=1, \ldots, n$ with vectors of numerical or categorical features 
$x_i \in \cX \subset \mathbb R^d$ and labels $y_i \in \cY$.
These correspond to the rows of `X` and the coordinates of `y` passed to `.fit(X, y)`.

## Random Forest

Given a vector of features $x \in \cX$, the prediction of a Random Forest is the average
\begin{equation*}
\widehat{g}(x ; \boldsymbol \Pi) = \frac 1M \sum_{m=1}^M \widehat{f}(x ; \Pi_m)
\end{equation*}
of the predictions of $M$ randomized decision trees $\widehat f(x ; \Pi_m)$ for 
$m=1, \ldots, M$ (following the principle of bagging {cite}`b-breiman1996bagging`) 
where $\Pi_m$ corresponds to random bootstrap and feature subsampling involved 
in their training (see {ref}`bootstrap` below).

These decision trees are trained independently of each other, in parallel, using 
different (independent and identically distributed realizations) stacked in 
$\boldsymbol \Pi = (\Pi_1, \ldots, \Pi_M)$. 
The number of trees $M$ corresponds to the `n_estimators` parameter in `WildWood` 
and defaults to $10$.
A first step involved in training `WildWood` is feature binning, as explained next.

### Feature binning

The split finding strategy described in {ref}`splits` below works on binned features. 
This technique is of common practice in extreme gradient boosting libraries 
{cite}`b-xgboost_paper, b-lightgbm_paper, b-prokhorenkova2017catboost` and we use the 
same approach in `WildWood`.
The input $n \times d$ matrix $\bs X$ of features is transformed into another 
same-size matrix of "binned" features denoted $\bs X^{\mathrm{bin}}$.
To each input feature $j=1, \ldots, d$ (columns of `X`) is associated a set 
$B_j = \{ 1, \ldots, b_j \}$ of bins, where $b_j \leq b_{\max}$ with $b_{\max}$ a 
hyperparameter called `max_bins` in `WildWood`, corresponding to the maximum number of 
bins a feature can use (default is `max_bins=256` similarly to {cite}`b-lightgbm_paper`, 
so that a single byte can be used for the entries of $\bs X^{\text {bin}}$).
When a feature is continuous, it is binned into $b_{\max}$ bins using inter-quantile intervals.
If it is categorical, each modality is mapped to a bin whenever $b_{\max}$ is larger than 
its number of modalities, otherwise the sparsest modalities end up binned together.
If a feature $j$ contains missing values, its rightmost bin in $B_j$ is used to encode them.
After binning, each column satisfies $\bs X_{\bullet, j}^\mathrm{bin} \in B_j^n$.

(tree)=
## Random decision trees

Let $\cell = \prod_{j=1}^d B_j$ be the binned feature space.
A random decision tree is a pair $(\tree, \splits)$, where $\tree$ is a finite ordered binary tree and $\splits$ contains information about each node in $\tree$, such as split information.
The tree is random and its source of randomness $\Pi$ comes from the bootstrap and feature subsampling as explained below.

### Finite ordered binary trees 

A finite ordered binary tree $\tree$ is represented as a finite subset of the set $\{ 0, 1 \}^* = \bigcup_{n \geq 0} \{ 0, 1 \}^n$ of all finite words on $\{ 0, 1\}$.
The set $\{ 0, 1\}^*$ is endowed with a tree structure (and called the complete binary tree): the empty word $\root$ is the root, and for any $\node \in \{ 0, 1\}^*$, the left (resp. right) child of $\node$ is $\node 0$ (resp. $\node 1$).
We denote by $\inodes (\tree) = \{ \node \in \tree : \node 0, \node 1 \in \tree \}$ the set of its interior nodes and by $\leaves (\tree) = \{ \node \in \tree : \node 0, \node 1 \not\in \tree \}$ the set of its leaves, both sets are disjoint and the set of all nodes is $\nodes(\tree) = \inodes (\tree) \cup \leaves (\tree)$.


### Splits and cells 

The split $\sigma_\node = (j_\node, t_\node) \in \Sigma$ of each $\node \in \inodes (\tree)$ is characterized by its dimension $j_\node \in \{ 1, \dots, d \}$ and a subset of bins $t_\node \subsetneq \{ 1, \ldots, b_{j_\node} \}$.
We associate to each $\node \in \tree$ a cell $\cell_\node \subseteq C$ which is defined recursively: $C_\root = C$ and for each $\node \in \inodes(\tree)$ we define
\begin{equation*}
  \cell_{\node 0} := \{ x \in \cell_\node : x_{j_\node} \in t_{\node}  \} \quad \text{and} \quad \cell_{\node 1} := \cell_\node \setminus \cell_{\node 0}.
\end{equation*}
When $j_\node$ corresponds to a continuous feature, bins have a natural order and $t_\node = \{ 1, 2, \ldots, s_{\node} \}$ for some bin threshold $s_{\node} \in B_{j_\node}$; 
while for a categorical split, the whole set $t_\node$ is required.
By construction, $(\cell_{\leaf})_{\leaf \in \leaves (\tree)}$ is a partition of $\cell$. 

(bootstrap)=
### Bootstrap and feature subsampling

Let $I = \{1, \ldots, n\}$ be the training samples indices.
The randomization $\Pi$ of the tree uses bootstrap: it samples uniformly at random, 
with replacement, elements of $I$ corresponding to in-the-bag ($\itb$) samples.
If we denote as $I_\itb$ the indices of unique $\itb$ samples, we can define the 
indices of out-of-bag ($\otb$) samples as $I_\otb = I \setminus I_\itb$.
A standard argument shows that $\P[i \in I_\itb] = 1 - (1 - 1/n)^n \rightarrow 1 - e^{-1} \approx 0.632$ 
as $n \rightarrow +\infty$, known as the 0.632 rule {cite}`b-efron1997improvements`.
The randomization $\Pi$ uses also feature subsampling: each time we need to find a split, 
we do not try all the features $\{1, \ldots, d\}$ but only a subset of them of size $d_{\max}$, 
chosen uniformly at random.
The $d_{\max}$ hyperparameter is called `max_features` in `WildWood`.
This follows what standard RF algorithms do 
{cite}`b-breiman2001randomforests, b-biau2016rf_tour, b-louppe2014understanding`, 
with the default $d_{\max} = \sqrt{d}$.

(splits)=
## Split finding on histograms

For $K$-class classification, when looking for a split for some node $\node$, we 
compute the node's "histogram"
\begin{equation*}
\mathrm{hist}_\node[j, b, k] = \sum_{i \in I_\itb : x_i \in \cell_\node} \ind{x_{i, j} = b, y_i = k}
\end{equation*}
for each sampled feature $j$, each bin $b$ and label class $k$ seen in the node's samples 
(actually weighted counts to handle bootstrapping and sample weights).
Of course, one has $\mathrm{hist}_\node = \mathrm{hist}_{\node 0} + \mathrm{hist}_{\node 1}$, so 
that we don't need to compute two histograms for siblings $\node 0$ and $\node 1$, but only a single one. 
Then, we loop over the set of non-constant (in the node) sampled features $\{ j : \# \{ b : \sum_{k} \mathrm{hist}_\node[j, b, k] \geq 1 \} \geq 2 \}$ 
and over the set of non-empty bins $\{ b : \sum_{k} \mathrm{hist}_\node[j, b, k] \geq 1 \}$ to find a split, by comparing standard 
impurity criteria computed on the histogram's statistics, such as gini or entropy for classification and variance for regression.

### Bin order and categorical features

The order of the bins used in the loop depends on the type of the feature.
If it is continuous, we use the natural order of bins.
If it is categorical and the task is binary classification (labels in $\{0, 1\}$) we use 
the bin order that sorts
\begin{equation*}
\frac{\mathrm{hist}_\node[j, b, 1]}{\sum_{k=0, 1} \mathrm{hist}_\node[j, b, k]}
\end{equation*}
with respect to $b$, namely the proportion of labels $1$ in each bin. 
This allows to find the optimal split with complexity $O(b_j \log b_j)$, see 
Theorem 9.6 in {cite}`b-breiman1984cart`, the logarithm coming from the sorting 
operation, while there are $2^{b_j - 1} -1$ possible splits.
This trick is used by extreme gradient boosting libraries as well, using an order of 
$\text{gradient} / \text{hessian}$ statistics of the loss considered 
{cite}`b-xgboost_paper, b-lightgbm_paper, b-prokhorenkova2017catboost`.

For $K$-class classification with $K > 2$, we consider two strategies: 

1. one-versus-rest, where we train $M K$ trees instead of $M$, each tree trained with a 
   binary one-versus-rest label, so that trees can find optimal categorical splits. 
   This corresponds to the `multiclass="ovr"` option in `WildWood`.
1. heuristic, where we train $M$ trees and where split finding uses $K$ loops over bin 
   orders that sort $\mathrm{hist}_\node[j, b, k] / \sum_{k'} \mathrm{hist}_\node[j, 
   b, k']$ (with respect to $b$) for $k=0, \ldots, K-1$. This corresponds to the 
   `multiclass="multinomial"` (default) and `cat_split_strategy="all"` options. 
   Note that `cat_split_strategy="binary"` and `cat_split_strategy="random"` are 
   also available, that respectively sort class 1 against the others and sort a 
   class selected at random against the others. 
   
If a feature contains missing values, we do not loop only left to right (along bin order), 
but right to left as well, in order to compare splits that put missing values on the left or on the right.

```{Warning}
The handling of missing values in `WildWood` is still under development.
```

### Split requirements

Nodes must hold at least one $\itb$ and one $\otb$ sample to apply aggregation with 
exponential weights, see {ref}`agg-ctw` below.
A split is discarded if it leads to children with less than `min_samples_leaf` 
$\itb$ or $\otb$ samples and we do not split a node with less than `min_samples_split` 
$\itb$ or $\otb$ samples.
These hyperparameters only weakly impact `WildWood`'s performances and sticking to 
default values 
(`min_samples_leaf=1` and `min_samples_split=2`, following `scikit-learn`'s defaults 
{cite}`b-louppe2014understanding, b-pedregosa2011scikit-learn`) is 
usually enough (see {cite}`b-wildwood` for experiments confirming this).

### Related works on categorical splits

In {cite}`b-PartitioningNominal`, an interesting characterization of an optimal 
categorical split for multiclass classification is introduced, but no efficient algorithm 
is, to the best of our understanding, available for it. 
A heuristic algorithm is proposed therein, but it requires to compute, for each split, the 
top principal component of the covariance matrix of the conditional distribution of labels 
given bins, which is computationally too demanding for a Random Forest algorithm intended 
for large datasets.
Regularized target encoding is shown in {cite}`b-pargent2021regularized` to perform 
best when compared with many alternative categorical encoding methods. 
Catboost {cite}`b-prokhorenkova2017catboost` uses target encoding, which replaces 
feature modalities by label statistics, so that a natural bin order can be used for split finding.
To avoid overfitting on uninformative categorical features, a debiasing technique uses 
random permutations of samples and computes the target statistic of each element based only 
on its predecessors in the permutation. 
However, for multiclass classification, target encoding is influenced by the arbitrarily 
chosen ordinal encoding of the labels.
LightGBM {cite}`b-lightgbm_paper` uses a one-versus-rest strategy, which is also one of 
the approaches used in `WildWood` for categorical splits on multiclass tasks.
For categorical splits, where bin order depends on labels statistics, `WildWood` does 
not use debiasing as in {cite}`b-prokhorenkova2017catboost`, since aggregation 
with exponential weights computed on $\otb$ samples allows to deal with overfitting.

### Tree growth stopping

We do not split a node and make it a leaf if it contains less than `min_samples_split` 
$\itb$ or $\otb$ samples.
When only leaves or non-splittable nodes remain, the growth of the tree is stopped.
Trees grow in a depth-first fashion so that childs $\node 0$ and $\node 1$ have memory 
indexes larger than their parent $\node$ (as required by Algorithm 1 below).


(agg-ctw)=
## Prediction function: aggregation with exponential weights

Given a tree $\tree$ grown as described in {ref}`tree` and {ref}`splits`, its prediction 
function is an aggregation of the predictions given by all possible subtrees rooted at 
$\root$, denoted $\{T : T \subset \tree \}$. 
While $\tree$ is grown using $\itb$ samples, we use $\otb$ samples to perform aggregation 
with exponential weights, with a branching process prior over subtrees, that gives more 
importance to subtrees with a good predictive $\otb$ performance.

### Node and subtree prediction

We define $\node_{T} (x) \in \leaves(T)$ as the leaf of $T$ containing $x \in \cell$.
The prediction of a node $\node \in \nodes(\tree)$ and of a subtree $T \subset \tree$ is 
given by
\begin{equation}
    \label{eq:node_subtree_prediction}
    \pred_{\node} = h ( (y_i)_{i \in I_\itb \pp x_i \in \cell_\node}) \quad \text{ and } 
    \quad  \pred_{T} (x) = \pred_{\node_{T} (x)},
\end{equation}
where $h : \cup_{n \geq 0} \cY^n \to \widehat \cY$ is a generic "forecaster" used in each 
cell and where a subtree prediction is the one of its leaf containing $x$.

A standard choice for regression ($\cY = \widehat \cY = \R$) is the empirical mean forecaster

$$
\pred_{\node} = \frac{1}{n_{\node}} \sum_{i \in I_\itb \pp x_i \in \cell_\node} y_i,
$$ (eqn:regpredictor)

where $n_{\node} = | \{i \in I_\itb \pp x_i \in \cell_\node \} |$.

For $K$-class classification with $\cY = \{ 1, \ldots, K \}$ and 
$\widehat \cY = \probas(\cY)$, the set of probability distributions over 
$\cY$, a standard choice is a Bayes predictive posterior with a prior on $\probas (\cY)$ 
equal to the Dirichlet distribution $\dirichletdist(\alpha, \dots, \alpha)$, namely 
the *Jeffreys prior* on the multinomial model $\probas (\cY)$, which leads to

$$
\pred_{\node} (k) = \frac{n_{\node} (k) + \alpha}{n_{\node} + \alpha K},
$$ (eqn:ktpredictor)

for any $k \in \cY$, where $n_{\node} (k) = | \{ i \in I_\itb : x_i \in \cell_\node, y_i = k \} |$.
By default, `WildWood` uses $\alpha = 1/2$ (the *Krichevsky-Trofimov* forecaster 
{cite}`b-tjalkens1993sequential`), but one can perfectly use any $\alpha > 0$, so that 
all the coordinates of $\pred_{\node}$ are positive.
The parameter $\alpha$ is called `dirichlet` in `WildWood`.
This is motivated by the fact that `WildWood` uses as default the log loss to assess $\otb$ performance 
for classification, which requires an arbitrarily chosen clipping value for zero probabilities.
Different choices of $\alpha$ only weakly impact `WildWood`'s performance, as illustrated in 
{cite}`b-wildwood`. 
We use $\otb$ samples to define the cumulative losses of the predictions of all $T \subset \tree$
\begin{equation}
    \label{eq:subtree_loss}
    L_T =  \sum_{i \in I_\otb} \ell (\pred_{T} (x_i), y_i),
\end{equation}
where $\loss : \widehat \cY \times \cY \to \R^+$ is a loss function.
For regression problems, a default choice is the quadratic loss 
$\ell (\pred, y) = (\pred - y)^2$ while for multiclass classification, a default is the 
log-loss $\ell (\pred, y) = - \log \pred(y)$, where $\pred(y) \in (0, 1]$ when using 
{eq}`eqn:ktpredictor`, but other loss choices are of course possible.

### Prediction function

Let $x \in \cell$.
The prediction function $\widehat f$ of a tree $\tree$ in `WildWood` is given by

$$
\widehat f (x) = \frac{\sum_{T \subset \tree} \pi (T) e^{-\eta L_T} \pred_{T} (x)}{\sum_{T \subset \tree} \pi (T) e^{-\eta L_T}} \quad \text{with} \quad \pi(T) = 2^{- \| T \|},
$$ (eq:exactaggregation)

where the sum is over all subtrees $T$ of $\tree$ rooted at $\root$, where $\eta > 
0$ is called `step` in `WildWood` and $\|T\|$ is the number of nodes in $T$ minus its 
number of leaves that are also leaves of $\tree$.
Note that $\pi$ is the distribution of the branching process with branching probability $1 / 2$ 
at each node of $\tree$, with exactly two children when it branches.
A default choice is `step=1.0` for the log-loss (as explained in {cite}`b-wildwood`), 
but it can also be tuned through hyperoptimization, although we do not observe strong 
performance gains {cite}`b-wildwood`.
The prediction function {eq}`eq:exactaggregation` is an aggregation of the predictions 
$\wh y_T(x)$ of all subtrees $T$ rooted at $\root$, weighted by their performance on 
$\otb$ samples. 
This aggregation procedure can be understood as a *non-greedy way to prune trees*: 
the weights depend not only on the quality of one single split but also on the performance of 
each subsequent split.

Computing $\widehat f$ from Equation {eq}`eq:exactaggregation` is computationally and 
memory-wise infeasible for a large $\tree$, since it involves a sum over all $T \subset \tree$ 
rooted at $\root$ and requires one weight for each $T$. 
Indeed, the number of subtrees of a minimal tree that separates $n$ points is 
*exponential* in the number of nodes, and hence *exponential* in $n$.
However, it turns out that one can compute exactly and very efficiently $\widehat f$ thanks 
to the prior choice $\pi$ together with an adaptation of *context tree weighting* 
{cite}`b-willems1995context-basic, b-willems1998context-extensions, b-helmbold1997pruning, b-catoni2004statistical`.

```{admonition} Theorem
The prediction function {eq}`eq:exactaggregation` can be written as 
$\wh f(x) = \wh f_{\root}(x)$, where $\wh f_{\root}(x)$ satisfies the recursion
\begin{equation}
    \label{eq:f_pred_recursion}
    \wh f_\node(x) = \frac 12 \frac{w_{\node}}{\wbar_\node} \pred_{\node} + \Big(1 - \frac 12  \frac{w_{\node}}{\wbar_\node} \Big) \wh f_{\node a}(x)
\end{equation}
for $\node, \node a \in \pathpoint(x)$ ($a \in \{0, 1\}$) the path in $\tree$ going from 
$\root$ to $\node_\tree(x)$, where $w_\node := \exp(-\eta L_\node)$ with 
$L_\node := \sum_{i \in I_\otb : x_i \in C_\node} \ell (\pred_{\node}, y_i)$ and where 
$\wbar_\node$ are weights satisfying the recursion
\begin{equation}
    \label{eq:wden-recursion}
    \wbar_{\node} =
    \begin{cases}
      w_{\node} & \text{ if } \node \in \leaves (\tree), \\
      \frac{1}{2} w_{\node} + \frac{1}{2} \wbar_{\node 0} \wbar_{\node 1} &\text{ otherwise}.
    \end{cases}
\end{equation}
```

The proof of this theorem is given in {cite}`b-wildwood`.
A consequence of it is a very efficient computation of $\wh f(x)$ as described in 
the Algorithms 1 and 2 below. 
Algorithm 1 computes the weights $\wbar_\node$ using the fact that trees in 
`WildWood` are grown in a depth-first fashion, so that we can loop *once*, leading 
to a $O(|\nodes(\tree)|)$ complexity in time and in memory usage, over nodes from a 
data structure that respects the parenthood order.
Direct computations can lead to numerical over- or under-flows (many products of exponentially 
small or large numbers are involved), so Algorithm 1 works recursively over the logarithms 
of the weights (line~6 uses a log-sum-exp function that can be made overflow-proof).

```{admonition} Algorithm 1 (Computation of $\log(\wbar_\node)$ for all $\node \in \nodes(\tree)$)

**Inputs:** 
: > $\tree$, $\eta > 0$ and losses $L_\node$ for all $\node \in \nodes(\tree)$. 
    Nodes from $\nodes(\tree)$ are stored in a data structure $\mathtt{nodes}$ that respects 
    parenthood order: for any $\node = \mathtt{nodes}[i_{\node}] \in \inodes(\tree)$ and children 
    $\node a = \mathtt{nodes}[i_{\node a}]$ for $a \in \{0, 1\}$, we have $i_{\node 
    a} > i_\node$. <br>

**for** $\node \in \mathrm{reversed}(\mathtt{nodes})$ <br>

: > **if** ($\node$ is a leaf) : Put $\log(\wbar_\node) \gets -\eta L_\node$<br> 
    **else** : Put  $\log(\wbar_{\node}) \gets \log( \frac{1}{2} e^{-\eta L_{\node}} + \frac {1}{2} e^{\log(\wbar_{\node 0}) + \log(\wbar_{\node 1})})$ <br>

**return** <br>
: > The set of log-weights $\{ \log(\wbar_{\node}) : \node \in \nodes (\tree) \}$
```

Algorithm 1 is applied once $\tree$ is fully grown, so that `WildWood` is ready to produce 
predictions using Algorithm 2 below.
Note that hyperoptimization of $\eta$ or $\alpha$ (`step` and `dirichlet` parameters 
in `WildWood`), if required, does not need to grow $\tree$ again, but only to update 
$\wbar_\node$ for all $\node \in \nodes(\tree)$ with Algorithm 1, making hyperoptimization 
of these parameters particularly efficient.

```{admonition} Algorithm 2 (Computation of $\wh f(x)$ for any $x \in C$)

**Inputs:** 
: > Tree $\tree$, losses $L_\node$ and log-weights $\log(\wbar_\node)$ computed by Algorithm 1

Find $\node_\tree(x) \in \leaves(\tree)$ (the leaf containing $x$) and put $\node 
\gets \node_\tree(x)$<br>
Put $\wh f(x) \gets \pred_\node$ (the node $\node$ forecaster, such as {eq}`eqn:regpredictor` 
for regression and {eq}`eqn:ktpredictor` for classification)<br>

**while** $\node \neq \root$ <br>
: > Put $\node \gets \mathrm{parent}(\node)$ <br>
    Put $\alpha \gets \frac 12 \exp(-\eta L_\node - \log(\wbar_\node))$ <br>
    Put $\wh f(x) \gets \alpha \pred_\node + (1 - \alpha) \wh f(x)$ <br>

**return**
: > The prediction $\wh f(x)$
```
The recursion used in Algorithm 2 has a complexity $O(|\pathpoint(x)|)$ which is the 
complexity required to find the leaf $\node_\tree(x)$ containing $x \in C$: 
Algorithm 2 *only increases by a factor $2$* the prediction complexity of a 
standard Random Forest (in order to go down to $\node_\tree(x)$ and up again to $\root$ 
along $\pathpoint(x)$).
More details about the construction of Algorithms 1 and 2 can be found in 
{cite}`b-wildwood`.

## References

```{bibliography} biblio.bib
---
labelprefix: B
keyprefix: b-
style: plain
filter: docname in docnames
---
