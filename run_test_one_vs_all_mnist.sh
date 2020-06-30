#!/bin/bash

set -e

mkdir -p './results_one_vs_all_mnist'
mkdir -p './models_one_vs_all_mnist'

for (( i = 0; i < 5; i++ )); do

  mkdir -p './models'

  python ./mnist_training.py ./models/mnist.h5

  mkdir -p './results'

  python ./multiclass_minim_one_vs_all_heuristic.py mnist.h5 [0.9,0.75,0.5,0.25,0.1,0.05] mnist

  mv ./results ./results_one_vs_all_mnist/results_simple_${i}

  mkdir -p './results'

  python ./multiclass_minim_one_vs_all_stable.py mnist.h5 [0.9,0.75,0.5,0.25,0.1,0.05] mnist

  mv ./results ./results_one_vs_all_mnist/results_extra_${i}

  mv ./models ./models_one_vs_all_mnist/models_${i}

done
