#!/usr/bin/env python

import os
import sys
import numpy as np
from keras import layers, losses
from keras import backend as K
from keras.models import load_model, Sequential
from keras.datasets import mnist, fashion_mnist
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm

def simplify_matrix(input_vertices_list, num_to_keep):
    """
    This function simplifies a neuron weight matrix, using the Heuristic Method (Algorithm 1).
    Arguments:
        input_vertices_list: Sorted list of input vertices
        num_to_keep: Number of neurons to keep in the final network.
    Returns:
        New array of neuron weights (and biases) for the hidden layer
    """

    # Failsafe for rounding errors.
    if(num_to_keep == 0):
        return np.zeros((0,input_vertices_list[0]))

    # Define output matrix
    output = np.zeros((num_to_keep, input_vertices_list[0].shape[0]))

    # Keep first vertex as first neuron
    output[0,:] = input_vertices_list[0]

    # For each new vertex, add difference with a previous neuron as a new neuron.
    # This is a simpler alternative to changing the previous neuron, since the
    # actual requirement is for their sum to be equal to the new vertex. This is
    # accomplished either as follows, or by setting the old neuron as the difference
    # and the new one as the new vertex.
    for i in range(1,min(num_to_keep, len(input_vertices_list))):
    	to_swap = np.random.randint(i)
    	output[i, :] = input_vertices_list[i] - output[to_swap,:]

    return output


def simplify_network_multiclass(oracle_model, dataset, labels, w1, b1, w2, b2, perc_to_keep_list):
    """
    This function simplifies the network provided, using the 1-vs-all method.
    Arguments:
        oracle_model: Model layers before the second-to-last.
        dataset: Training data used for the algorithm.
        labels: Training labels used for the method.
        w1: Weights of the second to last layer.
        b1: Biases of the second to last layer.
        w2: Weights of the output layer.
        b2: Biases of the output layer.
        perc_to_keep_list: Percentages to check.
    Returns:
        List of models, each with the desired percentage of neurons.
    """

    # Split in positive and negative part
    w2_pos = np.maximum(w2,0)
    w2_neg = -np.minimum(w2,0)

    num_neurons = w1.shape[0]
    num_classes = w2.shape[0]

    # Merge weights and biases into one matrix
    w1_full = np.concatenate([w1,b1.reshape((num_neurons,1))], axis=1)

    # Define neurons with positive and negative weights for each of the classes.
    neurons_pos = [w2_pos[i,:].reshape((-1,1)) * w1_full for i in range(num_classes)]
    neurons_neg = [w2_neg[i,:].reshape((-1,1)) * w1_full for i in range(num_classes)]

    counts_pos = [{} for i in range(num_classes)]
    counts_neg = [{} for i in range(num_classes)]

    for i in tqdm(range(dataset.shape[0])):
        # Use first layers to get represantation before FC
        d = oracle_model.predict(dataset[i,:].reshape(1,28,28,1))
        d_full = np.expand_dims(np.append(d, 1), axis=1)

        # For each of the classes, perform the binary classification minimization method.
        for l in range(num_classes):
            # Find activated neurons for each sample.
            activated_pos = np.maximum(np.dot(neurons_pos[l],d_full), 0).nonzero()[0]
            activated_neg = np.maximum(np.dot(neurons_neg[l],d_full), 0).nonzero()[0]

            # Here v_pos/v_neg are the activated vertices.
            v_pos = tuple(np.sum(neurons_pos[l][activated_pos, :], axis=0).tolist())
            v_neg = tuple(np.sum(neurons_neg[l][activated_neg, :], axis=0).tolist())

            # Assign a count, based on class label. We have 10 classes in this
            # case, so samples matching the label of the output neuron receive
            # a score of (num. of classes) - 1 = 9.
            if(l == labels[i]):
                counts_pos[l][v_pos] = counts_pos[l].get(v_pos, 0) + (num_classes-1)
                counts_neg[l][v_neg] = counts_neg[l].get(v_neg, 0) + (num_classes-1)
            else:
                counts_pos[l][v_pos] = counts_pos[l].get(v_pos, 0) + 1
                counts_neg[l][v_neg] = counts_neg[l].get(v_neg, 0) + 1

    # Get two new models, to calculate difference of activations between the original and the approximation.
    # Each has twice the number of classes as output neurons, to measure the positive and the negative
    # parts separately.
    orig_model_activ = Sequential()
    for layer in oracle_model.layers:
        orig_model_activ.add(layer)
    orig_model_activ.add(layers.Dense(w1.shape[0], activation='relu', weights=[w1.T, b1]))
    orig_model_activ.add(layers.Dense(2*w2.shape[0], activation='linear', use_bias=False, weights=[np.concatenate([w2_pos, w2_neg], axis=0).T]))
    predict_orig = orig_model_activ.predict(dataset)

    new_model_activ = Sequential()
    for layer in oracle_model.layers:
        new_model_activ.add(layer)
    new_model_activ.add(layers.Dense(w1.shape[0], activation='relu', weights=[w1.T, b1]))
    new_model_activ.add(layers.Dense(2*w2.shape[0], activation='linear', use_bias=False, weights=[np.concatenate([w2_pos, w2_neg], axis=0).T]))
    _ = new_model_activ.predict(dataset) # Make a dummy prediction to build the model.

    reduced_models = []
    for perc in perc_to_keep_list:
        print('Calculating percentage: '+str(perc))

        # For each class, essentially copy hidden layer and approximate each separately.
        new_pos_list = []
        new_neg_list = []
        for l in range(num_classes):
            new_pos_list.append(simplify_matrix([np.asarray(vert) for vert in sorted(counts_pos[l], key=counts_pos[l].get, reverse=True)], int(np.floor(0.1*perc*np.sum(w2_pos > 0, axis = 1)[l]))))
            new_neg_list.append(simplify_matrix([np.asarray(vert) for vert in sorted(counts_neg[l], key=counts_neg[l].get, reverse=True)], int(np.floor(0.1*perc*np.sum(w2_neg > 0, axis = 1)[l]))))

        # Define new neurons with pos/neg weights in the output layer (two parts of the hidden layer weight matrix)
        new_pos = np.concatenate(new_pos_list, axis = 0)
        new_neg = np.concatenate(new_neg_list, axis = 0)

        # Combine them into one new matrix
        w1_full_new = np.concatenate([new_pos, new_neg], axis=0)

        # In the output layer, assign weight +1 to all neurons with positive weight in the output layer as the original.
        # Same for negative, but with -1. w2_new is the weight matrix of the output layer in the final network, with
        # 10 classes in its output.
        w2_new = np.zeros((num_classes, w1_full_new.shape[0]))
        count = 0
        for l in range(num_classes):
            w2_new[l,count:count+new_pos_list[l].shape[0]] = 1
            count += new_pos_list[l].shape[0]
        for l in range(num_classes):
            w2_new[l,count:count+new_neg_list[l].shape[0]] = -1
            count += new_neg_list[l].shape[0]

        # Expand with zeros to a common size, to calculate activations
        w1_temp = np.pad(w1_full_new, ((0,num_neurons-w1_full_new.shape[0]),(0,0)))
        w2_temp = np.pad(w2_new, ((0,0),(0,num_neurons-w1_full_new.shape[0])))

        # Find activations in the new model (separating positive and negative parts in the output).
        new_model_activ.layers[-2].set_weights([w1_temp[:,:-1].T, w1_temp[:,-1]])
        new_model_activ.layers[-1].set_weights([np.concatenate([np.maximum(w2_temp, 0), -np.minimum(w2_temp,0)], axis =0).T])
        predict_new = new_model_activ.predict(dataset)

        # Calculate final bias, via the difference in average activations.
        final_bias = np.zeros(num_classes)
        for l in range(num_classes):
            # For each output neuron, find the activations for the samples only in the corresponding class
            dataset_extra = dataset[(labels == l),:]
            predict_orig_extra = orig_model_activ.predict(dataset_extra)
            predict_new_extra = new_model_activ.predict(dataset_extra)

            # Accumulate all predictions, before rebalancing the dataset by copying the activations with
            # label corresponding to this neuron.
            predict_new_full = predict_new
            predict_orig_full = predict_orig

            # For rebalancing, since we have (num of classes)-1 predictions corresponding to all other
            # labels, we add (num of classes)-2 = 8 copies of the predictions corresponding to the label
            # of this output neuron.
            for times in range(num_classes-2):
                predict_new_full = np.concatenate([predict_new_full, predict_new_extra])
                predict_orig_full = np.concatenate([predict_orig_full, predict_orig_extra])

            # Optimum bias of the second layer is found by the average differences in activations in the two parts
            optim_biases = np.mean(predict_orig-predict_new, axis=0)
            final_bias[l] = optim_biases[l]-optim_biases[l+num_classes]+b2[l] # Rejoin positive and negative parts.

        # Defined reduced model for this percentage, for this 10-class problem.
        reduced = Sequential()
        for layer in oracle_model.layers:
            reduced.add(layer)
        reduced.add(layers.Dense(w1_full_new.shape[0], activation='relu', weights=[w1_full_new[:,:-1].T, w1_full_new[:,-1]]))
        reduced.add(layers.Dense(num_classes, activation='softmax', weights=[w2_new.T, final_bias.flatten()]))
        reduced.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        reduced_models.append(reduced)

    return reduced_models

if __name__ == '__main__':

    if(len(sys.argv) != 4):
        print('Usage: multiclass_minim_one_vs_all.py <model_file> <[percentage1,...,percentageN]> <dataset>')
        exit()

    model_file = sys.argv[1]
    nums_to_check = [float(n) for n in sys.argv[2].strip('[]').split(',')]
    ds = sys.argv[3]

    # Load original model
    full_model = load_model(os.path.join('./models',model_file), compile=False)
    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # Load appropriate dataset
    if(ds == 'fashion_mnist'):
        (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
        train_data = train_data.reshape((train_data.shape[0], 28, 28, 1)).astype('float32')/255
        test_data = test_data.reshape((test_data.shape[0], 28, 28, 1)).astype('float32')/255
    elif(ds == 'mnist'):
        (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        train_data = train_data.reshape((train_data.shape[0], 28, 28, 1)).astype('float32')/255
        test_data = test_data.reshape((test_data.shape[0], 28, 28, 1)).astype('float32')/255
    else:
        raise NotImplementedError('Dataset not understood.')

    # Seed the generator.
    np.random.seed()

    # Evaluate original model.
    test_labels = to_categorical(test_labels)
    results_full = full_model.evaluate(test_data, test_labels)

    # Keep as oracle all layers but the last two.
    oracle_model = Sequential()
    for layer in full_model.layers[:-2]:
        oracle_model.add(layer)

    # Keep last two layers separately.
    w1_orig = full_model.layers[-2].get_weights()[0].T
    b1_orig = full_model.layers[-2].get_weights()[1]
    w2_orig = full_model.layers[-1].get_weights()[0].T
    b2_orig = full_model.layers[-1].get_weights()[1]

    # Perform minimization method on chosen percentages
    reduced_models = []
    reduced_models = simplify_network_multiclass(oracle_model, train_data, train_labels, w1_orig, b1_orig, w2_orig, b2_orig, nums_to_check)

    # Evaluate results and save to output.
    results = np.zeros((len(nums_to_check)+1, 2))
    results[0,0] = 1.0
    results[0,1] = results_full[1]
    for i, red in enumerate(reduced_models):
        n = nums_to_check[i]
        res = red.evaluate(test_data, test_labels, verbose=1)
        print('Accuracy when keeping {0:.1f}% of neurons: {1:.4f}'.format(n*100, res[1]))
        red.save('./models/{0:}_reduced_{1:.3f}.h5'.format(model_file.split('.')[0], n))
        results[i+1,0] = n
        results[i+1,1] = res[1]

    np.savetxt('./results/{0:}_{1:}.csv'.format(model_file.split('.')[0], '-'.join([str(n) for n in nums_to_check])), results, fmt='%.5f', delimiter=',')
