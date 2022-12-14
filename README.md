<h1>Recsys Notebooks</h1>
<p>A compilation of recommender system benchmarks.</p>
<br>
<h2>Implementation Details</h2>

The primary plan is to utilize multiple CPU cores as much as possible and disregard the use of GPUs. Data preprocessing is done mostly with [Pyspark](https://spark.apache.org/docs/latest/api/python/) when handling extremely large datasets. To minimize computation time, Numpy is preferred over PyTorch or Tensorflow, and is JIT compiled with [Numba](http://numba.pydata.org/) whenever possible.
<br>
<h2>List of implementations</h2>
<h3>1. Matrix Factorization</h3>

Reproduction of the paper [Matrix Factorization for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf). I implement from the paper equation (2), which is the pure matrix factorization with regularization(lambda=0.1), without biases or external information. I use momentum(p=0.9) to speed up stochastic gradient descent(lr=0.01). The algorithm seems to converge after 1 epoch of training(210 seconds).
|            | Original Paper | Reproduced |
|:----------:|:--------------:|:----------:|
| **RMSE(k=40)** |      0.91      |    1.08    |
