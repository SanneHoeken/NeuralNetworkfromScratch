import numpy as np
from datasets import load_dataset
from itertools import chain
from gensim.models import KeyedVectors
import pickle

def preprocess_conll_data(dataset, embedding_model, dimensions):
    """
    Docstring
    """
    tokens = list(chain.from_iterable(dataset['tokens']))
    pos_tags = list(chain.from_iterable(dataset['pos_tags']))

    # transform tokens to embeddings
    inputs_list = []
    for token in tokens:
        if token in embedding_model:
            vector = embedding_model[token]
        else:
            vector = [0]*dimensions
        inputs_list.append(vector)

    return inputs_list, pos_tags


def create_output_values(targets, output_nodes, activation_function='sigmoid'):
    """
    Docstring
    """
    targets_output_list = []
    for target in targets:
        if activation_function == 'sigmoid':
            # create an array filled with 0.01's
            target_value = np.zeros(output_nodes) + 0.01
        elif activation_function == 'tanh':
            # create an array filled with -0.99's
            target_value = np.zeros(output_nodes) - 0.99
        # tag label is the index of the array that is set to 0.99
        target_value[target] = 0.99
        targets_output_list.append(target_value)

    return targets_output_list

def main():

    output_nodes = 12

    # preprocess the training data
    embedding_model = KeyedVectors.load_word2vec_format("data/cow-embeddings-320/cow-big.txt", binary=False)
    train_dataset = load_dataset("conll2002", "nl", split='train')
    inputs_list, targets_list = preprocess_conll_data(train_dataset, embedding_model, 320)
    pickle.dump(inputs_list, open('./data/conll2002/train_inputs.txt', 'wb'))
    
    #create output values for sigmoid as activation function
    targets_output_list = create_output_values(targets_list, output_nodes)
    pickle.dump(targets_output_list, open('./data/conll2002/train_targets_output.txt', 'wb'))
    
    # create output values for tanh as activation function
    targets_output_list = create_output_values(targets_list, output_nodes, activation_function='tanh')
    pickle.dump(targets_output_list, open('./data/conll2002/train_targets_tanh_output.txt', 'wb'))
    
    # preprocess the validation data
    validation_dataset = load_dataset("conll2002", "nl", split='validation')
    inputs_list, targets_list = preprocess_conll_data(validation_dataset, embedding_model, 320)

    pickle.dump(inputs_list, open('./data/conll2002/validation_inputs.txt', 'wb'))
    pickle.dump(targets_list, open('./data/conll2002/validation_targets.txt', 'wb'))
    
    # preprocess the test data
    test_dataset = load_dataset("conll2002", "nl", split='test')
    inputs_list, targets_list = preprocess_conll_data(test_dataset, embedding_model, 320)

    pickle.dump(inputs_list, open('./data/conll2002/test_inputs.txt', 'wb'))
    pickle.dump(targets_list, open('./data/conll2002/test_targets.txt', 'wb'))

if __name__ == "__main__":
    main()