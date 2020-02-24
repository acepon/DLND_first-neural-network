#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def p(a, name, printable = False):
    if printable:
        print('-'*(len(name) + 15))
        print(f'{name}.shape: {a.shape}')
        print(a)
    else:
        pass

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, dropout = 0):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x : 1/(1 + np.exp(-x))
        self.activation_prime = lambda x: self.activation_function(x) * (1.0 - self.activation_function(x))

        if dropout is None:
            self._dropout = 0.0
        else:
            self._dropout = dropout

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            p(X, 'X')
            p(y, 'y')
            ### Forward pass ###
            
            # X ~ (m,)
            # self.weights_input_to_hidden ~ (m, h)
            # self.weights_hidden_to_output ~ (h, o)

            # hidden_inputs is the 
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            p(hidden_inputs, 'hidden_inputs')
            # hidden_inputs = (m,) . (m, h) = {h,}
            
            hidden_outputs = self.activation_function(hidden_inputs)
            p(hidden_outputs, 'hidden_outputs')
            
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            p(final_inputs, 'final_inputs')
            # final_inputs = （h,) . (h, o) = (o, )
            
            final_outputs = final_inputs
            
            ### Backward pass ###
            
            # calculating y - y_hat
            error = y - final_outputs
            p(error, 'error')
            
            # the d lambda x: x / d x is 1
            output_error_term = error * 1.0
            p(output_error_term, 'output_error_term')
            
            # weight the error back to hidden layer
            hidden_error = np.dot(self.weights_hidden_to_output, error) 
            p(hidden_error, 'hidden_error')
            # hidden_error = (h, o) . (o,) = (h,)
            
            # chain rule: error * sigmoid_prime to retrive back hidden_inputs value
            hidden_error_term = hidden_error * self.activation_prime(hidden_inputs)
            p(hidden_error_term, 'hidden_error_term')
            
            # chain rule: error * sigmoid_prime * features to retrive back weight0
            delta_weights_i_h += hidden_error_term * X[:,None]
            p(delta_weights_i_h, 'delta_weights_i_h')
            # delta_weights_i_h = (h,) . (m, 1) = (m, h)
            
            # chain rule: error * 1 * features to retrive back weight1
            delta_weights_h_o += output_error_term * hidden_outputs[:,None]
            p(delta_weights_h_o, 'delta_weights_h_o')
            # delta_weights_h_o = (o,) . (h,1) = (h, o)

            
        # Weights update
        # dropout implementation
        weights_hidden_to_output_drop = np.random.binomial(1, 1-self._dropout, self.weights_hidden_to_output.shape)
        weights_input_to_hidden_drop = np.random.binomial(1, 1-self._dropout, self.weights_input_to_hidden.shape)

        self.weights_hidden_to_output += self.lr*delta_weights_h_o
        self.weights_hidden_to_output *= weights_hidden_to_output_drop
        self.weights_input_to_hidden += self.lr*delta_weights_i_h
        self.weights_input_to_hidden *= weights_input_to_hidden_drop
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Forward pass
        hidden_inputs =  np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs 
        return final_outputs
