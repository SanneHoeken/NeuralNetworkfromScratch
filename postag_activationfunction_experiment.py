import numpy as np
import pickle, csv, time
from sklearn.metrics import accuracy_score
from neural_network import neuralNetwork

def main():

    # get training data
    train_inputs_list = pickle.load(open('./data/conll2002/train_inputs.txt', 'rb'))
    train_targets_output_list = pickle.load(open('./data/conll2002/train_targets_output.txt', 'rb'))

    # get validation data
    val_inputs_list = pickle.load(open('./data/conll2002/validation_inputs.txt', 'rb'))
    val_targets_list = pickle.load(open('./data/conll2002/validation_targets.txt', 'rb'))

    # set parameters of the network
    input_nodes = 320
    output_nodes = 12
    hidden_nodes = 100
    learning_rate = 0.3
    epochs = 3

    results = []

    # iterate over 3 times over every activation function
    for i in range(3):
        for activation_function in ('sigmoid', 'tanh'):
                
                # store the choice of activation function as a result dictionary
                result = {'activation_function': activation_function}

                # create instance of neural network
                n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate, activation_function)

                # train the neural network and measure the time it takes
                start = time.perf_counter()
                n.train(train_inputs_list, train_targets_output_list, epochs)
                stop = time.perf_counter()

                # add training time to result
                result['time'] = stop - start

                # query predictions for the validation data
                predictions = []
                for inp in val_inputs_list:
                    # query the network
                    output = n.forward(np.array(inp, ndmin=2).T)[0]
                    # label is index with highest value 
                    predictions.append(np.argmax(output))

                # add accuracy score to result and append to list of results
                result['accuracy'] = accuracy_score(val_targets_list, predictions)
                results.append(result)

    # save results as csv-file
    with open("postag_activationfunction_experiments.csv", 'w', newline='') as outfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    

if __name__ == "__main__":
    main()