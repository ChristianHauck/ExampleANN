#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

from layer import Layer
from functions import Functions
import numpy as np
import math


class MLP:

    def __init__(self, dimensions): # first element = input-dim, the following = layer-dims
        self.num_layers = len(dimensions) - 1   # -1 : subtract input-dim
        self.layers = []
        # and now the specified layers
        for i in range(self.num_layers):
            num_in = dimensions[i]
            num_out = dimensions[i+1]
            self.layers.append(Layer(num_in, num_out))
        # activation functions
        self.activation_str = None
        self.functions = Functions()
        return


    def init_layers(self):
        for layer in self.layers:
            rnd_wght = np.random.rand(layer.num_out, layer.num_in) / 1
            layer.init_weights(rnd_wght)


    def set_activation(self, activation_str):
        self.activation_str = activation_str


    def activation(self, vector):
        result = self.functions.get_fu[self.activation_str](vector)
        return result


    def deriv_act(self, vector):
        result = self.functions.get_deriv[self.activation_str](vector)
        return result


    def learn(self, m_x, m_y, learning_rate, max_loops=1000):
        loop_nr = 0
        while True:
            for i in range(len(m_x)):                  # for each example - index: i
                # Just copy the input into INPUT of layer 0
                self.layers[0].input = m_x[i]

                # Now propagate a[0] forward and store the interim results a[l] for the layers l
                # for l in range(self.num_layers):    # for each layer except of the first - index l
                #     layer = self.layers[l]
                #     y = np.empty(layer.num_out)
                #     res = np.empty(layer.num_out)
                #     for j in range(layer.num_out):    # for each row in weight matrix of layer l - index j
                #         res[j] = np.dot(layer.weights[j], layer.input)
                #         y[j] = self.activation(res[j])
                #
                #     layer.sum = res   # = SUM(wv)
                #     layer.output = y  # = g(RAW_OUT) = g(SUM(wv))
                #     if l < self.num_layers - 1:
                #         self.layers[l + 1].input = np.copy(self.layers[l].output)

                for l in range(self.num_layers):
                    self.layers[l].sum = np.dot(self.layers[l].weights, self.layers[l].input)
                    self.layers[l].output = self.activation(self.layers[l].sum)
                    if (l < self.num_layers - 1):
                        self.layers[l + 1].input = self.layers[l].output

                # Compute the error at the output layer
                out_layer = self.layers[self.num_layers - 1]  # output layer
                out_layer.delta = self.deriv_act(out_layer.output) * (m_y[i] - out_layer.output) # i = example index

                # Compute the other deltas, backwards running through the layers
                for l in reversed(range(self.num_layers - 1)):
                    layer_m = self.layers[l]
                    layer_m_plus_1 = self.layers[l+1]
                    delta_m = self.deriv_act(layer_m.sum) * \
                              np.dot(layer_m_plus_1.weights.T, layer_m_plus_1.delta)
                    layer_m.delta = delta_m

                # And finally adjust weights.
                # This may be done in forward order because all data are available here.
                for l in range(self.num_layers):
                    layer = self.layers[l]
                    for j in range(layer.num_out):
                        delta_w = learning_rate * layer.delta[j] * layer.input
                        row = layer.weights[j] + delta_w
                        layer.weights[j] = row
                        # layer.weights[j] = layer.weights[j] + delta_w

            # Calculate a more or less reasonable error metric
            error = Functions.squared_error(m_y[i], self.layers[self.num_layers - 1].output)
            error = math.sqrt(error)
            loop_nr += 1

            if error < 0.001 or loop_nr > max_loops:
                # print("learning loops: ", loop_nr)
                break  # while

        return


    def run(self, x):
        self.layers[0].input = x
        for l in range(self.num_layers):
            self.layers[l].sum = np.dot(self.layers[l].weights, self.layers[l].input)
            # self.layers[l].output = self.activation(self.layers[l].sum)
            self.layers[l].output = self.activation(self.layers[l].sum)
            if (l < self.num_layers - 1):
                self.layers[l+1].input = self.layers[l].output
        #
        return self.layers[self.num_layers - 1].output


    def show(self, out_str, m_x, m_y):
        print(out_str)
        # for i in range(self.num_layers):
        #     print("weigt matrix: ", self.layers[i].weights)
        for i in range(len(m_x)):
            result = self.run(m_x[i])
            # print("input ", m_x[i], "desired output ", m_y[i], " output ", result)
            print("desired output ", m_y[i])
            print("output ", result)


if __name__ == '__main__':
    mlp = MLP([3,5,2])
    mlp.init_layers()
    mlp.init_activation("sigmoid")
