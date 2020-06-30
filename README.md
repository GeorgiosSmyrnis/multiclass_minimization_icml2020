# Multiclass Neural Network Minimization via Tropical Newton Polytope Approximation
## Georgios Smyrnis, Petros Maragos

This is the repository for the code of our ICML 2020 paper, "Multiclass Neural Network Minimization via Tropical Newton Polytope Approximation". The provided code produces the results for the experiments, as defined in the paper. Moreover, a script is also provided, which creates the figures presented in the paper.

## Citation

If you use this code, please cite the following:
```bibtex
@InProceedings{SmyrnisMaragos_2020_ICML,
author = {Smyrnis, Georgios and Maragos, Petros},
title = {Multiclass Neural Network Minimization via Tropical Newton Polytope Approximation},
booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
publisher = {PMLR},
month = {July},
year = {2020}
} 
```

## Datasets Used

For our experiments we used the following datasets. We encourage anyone interested to visit the original sources and cite the appropriate references if they make use of these datasets.

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## File Description:
The following files are contained:
  - `mnist_training.py`: Trains the base model to be minimized, for the MNIST
    dataset.
  - `fashion_mnist_training.py`: Trains the base model to be minimized, for the
    Fashion-MNIST dataset.
  - `multiclass_minim_one_vs_all_heuristic.py`: Performs the One-Vs-All multiclass minimization procedure,
    outlined in Section 4.2, based on the single output minimizaton algorithm from Section 3.2.
  - `multiclass_minim_one_vs_all_stable.py`: Performs the same minimization procedure, but
    using as single output minimization algorithm the one from Section 5.1.
  - `result_aggregation_mnist.py`: Aggregates results from the experimental runs,
    calculating mean and standard deviation (note that the same code is used
    for both datasets, see guidelines below).
  - `create_figures.py`: Creates the figures presented in the paper.
  - `run_test_one_vs_all_mnist.sh`: Script which runs the experiments for MNIST.
  - `run_test_one_vs_all_fashion.sh`: Script which runs the experiments for
    Fashion-MNIST.


## Experimental Section:
For the MNIST experiments (Tables 1, 2), run the following three commands:
  - `./run_test_one_vs_all.sh`
  - `python result_aggregation_mnist.py ./results_one_vs_all_mnist simple`
  - `python result_aggregation_mnist.py ./results_one_vs_all_mnist extra`

For the Fashion-MNIST experiments (Table 3), run the following two commands:
  - `./run_test_one_vs_all_fashion.sh`
  - `python result_aggregation_mnist.py ./results_one_vs_all_fashion extra`

The results can be found as follows ("simple" corresponds to the method of
Section 4.2 with the Heuristic Minimization, and "extra" to the use of the
Stable Minimization of Section 5.1, as explained in the experimental section
of the paper):
  - Table 1: `./results_one_vs_all_mnist/results_simple/`
  - Table 2: `./results_one_vs_all_mnist/results_extra/`
  - Table 3: `./results_one_vs_all_fashion/results_extra/`

## Output format:
For the files without the `_std` mark:
  - Column 1: Percentages.
  - Column 2: Average Score.

For the files with the `_std` mark:
  - Column 2: Standard deviation (for the percentage corresponding to each row).

Note that the folders named `results_<simple/extra>_<N>` contain results from
the individual trials.


## Figures:
For the figures, run the following command (each figure will be presented and
can then be saved manually, in the desired format):

  `python create_figures.py`


## Further Details - Requirements:
These experiments were run using Python 3.7.6, on an Ubuntu 18.04 OS.
The following packages were installed on the testing environment and used in the provided code:
  - numpy:        v1.18.1
  - scipy:        v1.3.2
  - tensorflow:   v1.15.0
  - keras:        v2.2.4
  - tqdm:         v4.41.1
  - pandas:       v0.25.3
  - matplotlib:   v3.1.3

All of the dependencies of the above were also installed in the testing environment, as well as scikit-learn, v0.22.1.
Installing these packages, along with their respective dependencies, leads to
an environment equivalent to the one on which the experiments were run. Note
that this is not a strict requirement; it might be possible to run the above
experiments in a different configuration.

## References
[1] G.Smyrnis and P. Maragos. Multiclass Neural Network Minimization via Tropical Newton Polytope Approximation. ICML 2020.

[2] [F. Chollet et al. Keras. 2015.](https://keras.io)

[3] [Y. Lecun, L. Bottou, Y. Bengio and P. Haffner. Gradient-Based Learning Applied to Document Recognition. Proc. of the IEEE 1998.](https://ieeexplore.ieee.org/abstract/document/726791) 

[4] [H. Xiao, K. Rasul and R. Vollgraf. Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv 2017.](https://arxiv.org/abs/1708.07747)

## License

Our code is released under the MIT license.
