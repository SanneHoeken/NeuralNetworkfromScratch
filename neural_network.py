import numpy as np

class neuralNetwork:
    """
    The neuralNetwork class that initializes a simple one-layer neural netwerk
    with specified number of input, hidden and output nodes, learning rate and activation function
    """

    def __init__(self, inputnodes, outputnodes, hiddennodes, learningrate, activation_function):
        
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # sampling weight matrixes from normal distribution with mean of zero 
        # and standard deviation of 1/√(number of incoming links)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # set learning rate
        self.lr = learningrate

        # set activation function and its derivative
        self.activation_function = self.get_activation_function(activation_function)
        self.activation_derivative = self.get_activation_derivative(activation_function)
    
    
    def get_activation_function(self, activation_function):
        """
        Returns the sigmoid or hyperbolic tangent function as activation function
        """
        if activation_function == 'sigmoid':
            # return sigmoid function 
            return lambda x: 1 / (1 + np.exp(-x))

        elif activation_function == 'tanh':
            # return hyperbolic tangent (tanh) 
            return lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


    def get_activation_derivative(self, activation_function):
        """
        Returns the derivative of the sigmoid or hyperbolic tangent function
        """
        if activation_function == 'sigmoid':
            # return derivative of sigmoid function 
            return lambda x: self.activation_function(x)*(1.0 - self.activation_function(x))

        elif activation_function == 'tanh':
            # set hyperbolic tangent (tanh) as activation function
            return lambda x: 1 - (self.activation_function(x))**2


    def forward(self, inputs):
        """
        Takes the input to the neural network and returns the network's output
        """
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs, final_inputs, hidden_outputs, hidden_inputs


    def backward(self, inputs, targets):
        """
        Refines the network's weights at each layer by
        computing output for given training examples,
        comparing this output with the desired output
        and using the difference to guide the updating of the weights
        """
        # get hidden and final outputs
        final_outputs, final_inputs, hidden_outputs, hidden_inputs = self.forward(inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.activation_derivative(final_inputs)), np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * self.activation_derivative(hidden_inputs)), np.transpose(inputs))
    
    
    def train(self, inputs_list, targets_list, epochs):
        """
        Iteratively updates the network's weights for every pair
        of observation (from input_list) and target label (from target_list),
        and repeats this a specified number of epochs times.
        """
        for e in range(epochs):
            # iterate over all inputs and targets
            for inp, targ in zip(inputs_list, targets_list):
                
                # convert input and target to 2d array
                inp = np.array(inp, ndmin=2).T
                targ = np.array(targ, ndmin=2).T

                # train the network
                self.backward(inp, targ)

    




