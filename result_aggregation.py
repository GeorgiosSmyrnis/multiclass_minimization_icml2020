#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import os
import shutil

if __name__ == '__main__':
    """
    Aggregate results produced by the program.
    For the ouput format see README.txt.
    """

    basedir = sys.argv[1]
    type = sys.argv[2] # extra or simple

    # Remove old results if they exist
    extradir = 'results_'+type
    if(os.path.exists(os.path.join(basedir, extradir))):
        shutil.rmtree(os.path.join(basedir, extradir))

    results_full = []
    for dir in os.listdir(basedir):

        if(type not in dir):
            continue

        # Accumulate results for trial (in case they are split in multiple files).
        results = []
        for file in os.listdir(os.path.join(basedir, dir)):
            df = pd.read_csv(os.path.join(basedir, dir, file), header=None)
            results.append(df.values)

        arr = np.concatenate(results, axis = 0)
        arr = arr[arr[:,0].argsort()[::-1]] # Sort from highest percentage to lowest
        results_full.append(arr)

    # Save result mean and std.
    results_mean = sum(results_full)/len(results_full)
    results_dev = np.std(np.concatenate([np.expand_dims(m,axis=0) for m in results_full], axis=0), axis=0)

    # Save to output file.
    os.mkdir(os.path.join(basedir, extradir))
    np.savetxt(os.path.join(basedir, extradir, 'full_results.csv'), results_mean, fmt='%.5f', delimiter=',')
    np.savetxt(os.path.join(basedir, extradir, 'full_results_std.csv'), results_dev, fmt='%.5f', delimiter=',')
