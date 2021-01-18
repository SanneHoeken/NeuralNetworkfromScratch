import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from own_neural_network import neuralNetwork

def preprocess_mnist_data(filepath, output_nodes):
    """
    docstring
    """
    # load the mnist training data csv file into a dataframe
    data = pd.read_csv(filepath, header=None)
    # extract inputs and convert all values to float
    input_df = data.drop(0, axis=1).astype(float) 
    # scale and shift the inputs
    scaled_input = (input_df/255.0 * 0.99) + 0.01
    # convert inputs to list
    inputs_list = scaled_input.values.tolist()
    # extract the target labels
    targets_list = data[0].to_list()
    # create the target output values
    targets_output_list = []
    # iterate over every target label
    for target in targets_list:
        # create an array filled with 0.01's
        target_value = np.zeros(output_nodes) + 0.01
        # target label is the index of the array that is set to 0.99
        target_value[target] = 0.99
        targets_output_list.append(target_value)

    return inputs_list, targets_list, targets_output_list


def main():

    # initialize number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # initialize learning rate
    learning_rate = 0.1

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # set number of times the training data is used for training
    epochs = 5

    # preprocess the training data
    training_data_file = "MNIST_Dataset/mnist_train_100.csv"
    inputs_list, targets_list, targets_output_list = preprocess_mnist_data(training_data_file, output_nodes)
    
    # train the neural network
    n.train(inputs_list, targets_output_list, epochs)
    
    # preprocess the test data
    test_data_file = "MNIST_Dataset/mnist_test_10.csv"
    inputs_list, targets_list, targets_output_list = preprocess_mnist_data(test_data_file, output_nodes)
    
    # query predictions for the test data
    predictions = []
    for inp in inputs_list:
        # query the network
        output = n.forward(np.array(inp, ndmin=2).T)[0]
        # label is index with highest value 
        predictions.append(np.argmax(output))

    # provide classification report
    print(classification_report(targets_list, predictions))
    

if __name__ == "__main__":
    main()