#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

from layer2 import Layer2
from mlp import MLP
from mlp2 import MLP2
from utils import Utils
import numpy as np
import time


class Program:

    def __init__(self):
        self.m_x = None
        self.m_y = None


    def set_input_1(self):
        self.m_x = np.array([
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, 1, 1]
        ])
        self.m_y = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.6, 0.6],
            [0.9, 0.8],
        ])
        return


    def test1(self, mlp, learn_rate, num_loops):
        mlp.show("before training", self.m_x, self.m_y)  # before training
        # training
        mlp.learn(
            self.m_x,       # input matrix/vectors
            self.m_y,   # output target vector of "or" function
            learn_rate, # learning rate
            num_loops   # number of loops
            )
        #
        mlp.show("after training", self.m_x, self.m_y)  # after training
        #
        # Now test an input vector unknown to the net
        arr = np.array([-1, 0.8, 0.7])
        print('Testing network with input ', arr, ' : ', mlp.run(arr))
        #
        return


    def test2(self, mlp, learn_rate, num_loops):
        mlp.learn(
            self.m_x,       # input matrix/vectors
            self.m_y,   # output target vector of "or" function
            learn_rate, # learning rate
            num_loops   # number of loops
            )
        return


    def benchmark(self):
        # Single Perceptron-Layer
        layer = Layer2(num_in=4, num_out=2)
        layer.input = np.array([0, 0, 0, 1])
        layer.weights = np.random.rand(4, 2)
        layer.run()

        rounds = 1000
        tick_down = 1000
        # Multi-Layer-Perceptron
        mlp = MLP2([3, 5, 2])
        mlp.init_layers()
        mlp.set_activation("sigmoid")
        # Test
        self.set_input_1()
        start_time = time.time()
        for i in range(rounds):
            self.test2(mlp, 10, 1000)
        elapsed_1 = time.time() - start_time
        print "numpy elapsed time: ", elapsed_1, "s"

        # Partly numpy version
        mlp = MLP([3, 5, 2])
        mlp.test_encoder(10, 1000)
        # Test
        start_time = time.time()
        for i in range(rounds / tick_down):
            mlp.test_encoder(10, 1000)
        elapsed_2 = time.time() - start_time
        print "mixed elapsed time: ", elapsed_2, "s"

        print "speed factor: ", elapsed_2 / elapsed_1 * tick_down


if __name__ == '__main__':
    prog = Program()
    # prog.benchmark()
    num_items, rows, columns, data = Utils.load_mnist_data(Utils.mnist_train_data)
    print "num_items=", num_items, " rows=", rows, " columns=", columns
    print data[0]
