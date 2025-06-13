import numpy as np


## Numpy methods:
## exp() generates natural exponential
## array() generates a matrix
## dot() also dot multiplication of matrices
## random() generates random numbers. Need to be seeded for efficient distribution.


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1) #sets seed for random number generation

    ## np.random.random(size = none):
        ## Return random floats in the half-open interval [0.0, 1.0).
        ## Results are from the “continuous uniform” distribution over the stated interval.
        ## To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
        ## (b - a) * random_sample() + a
        ## To convert weights to a 3x1 matrix with values from -1 to 1, mean of 0:
            ## (1 - -1) * np.random.random((3, 1)) + -1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        

    def sigmoid(self, x):
        ## sigmoid function is o(z): 1 / 1 + e^-z
        ## this will take sigmoid function of the input data
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # Train model to make predictions while continuously adjusting weights
        for iteration in range(training_iterations):
            # Siphon the training data via neuron
            output = self.think(training_inputs)

            # Compute the error rate for back-propogation
            error = training_outputs - output

            # Perform weight adjustments
                # .T is the transposed matrix
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # pass in the inputs from neuron to get output
        # convert values to floats
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":
    # initialize neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights:")
    print(neural_network.synaptic_weights)

    # training data consists of 4 examples-- 3 input values and 1 ouput
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    # perform training
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Weights After Training:")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: ", user_input_one,
          user_input_two, user_input_three)
    print("New Output Data: ")
    print(neural_network.think(np.array([user_input_one,
                                         user_input_two,
                                         user_input_three])))
    print("Process Complete")
