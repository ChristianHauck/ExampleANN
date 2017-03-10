#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>

import numpy as np
from layer import Layer
from functions import Functions


class SLP():    # Single Layer Perceptron

    def __init__(self, in_nodes, layer_nodes):
        self.num_in = in_nodes
        self.num_nodes = layer_nodes
        self.layer = Layer(in_nodes, layer_nodes, 1) # 1 == threshold
        self.activation = Functions.tanh


    def test_sigmoid(self):
        layer = Layer(3, 2, threshold=0.0, activation='sigmoid')
        M_in = np.array([
            [-1,0,0],
            [-1,0,1],
            [-1,1,0],
            [-1,1,1]
            #[-1, 1, 1, 1, 1, 1, 1, 0],  # 0
            #[-1, 0, 1, 1, 0, 0, 0, 0],  # 1
            #[-1, 1, 1, 0, 1, 1, 0, 1],  # 2
            #[-1, 1, 0, 0, 1, 1, 1, 1],  # 3
            #[-1, 0, 1, 1, 0, 0, 1, 1],  # 4
            #[-1, 1, 0, 1, 1, 0, 1, 1],  # 5
            #[-1, 0, 0, 1, 1, 1, 1, 1],  # 6
            #[-1, 1, 1, 1, 0, 0, 0, 0],  # 7
            #[-1, 1, 1, 1, 1, 1, 1, 1],  # 8
            #[-1, 1, 1, 0, 0, 1, 1, 1]   # 9
            #[-1, 1, 1, 1, 1, 1, 1, 1, 0],  # 0
            #[-1, 1, 0, 1, 1, 0, 0, 0, 0],  # 1
            #[-1, 1, 1, 1, 0, 1, 1, 0, 1],  # 2
            #[-1, 2, 1, 0, 0, 1, 1, 1, 1],  # 3
            #[-1, 2, 0, 1, 1, 0, 0, 1, 1],  # 4
            #[-1, 2, 1, 0, 1, 1, 0, 1, 1],  # 5
            #[-1, 4, 0, 0, 1, 1, 1, 1, 1],  # 6
            #[-1, 4, 1, 1, 1, 0, 0, 0, 0],  # 7
            #[-1, 4, 1, 1, 1, 1, 1, 1, 1],  # 8
            #[-1, 6, 1, 1, 0, 0, 1, 1, 1]   # 9
            ])
        v_target = np.array([
            [0.1],
            [0.3],
            [0.6],
            [0.9],
            #[4],
            #[5],
            #[6],
            #[7],
            #[8],
            #[9]
            ])

        #m_weights = np.random.rand(layer.num_nodes, layer.num_in)
        m_weights = np.full((layer.num_nodes, layer.num_in), 0.0)
        layer.init_weights(m_weights)

        layer.show("before training", M_in, v_target)  # before training
        # training
        layer.learn_wo_thr(
            M_in,           # input matrix/vectors
            v_target,    # output target vector of "or" function
            5,            # learning rate
            )

        layer.show("after training", M_in, v_target)  # after training

        # Test an input vector unknown to the net
        arr = np.array([-1, 0.9, 0.1])
        print('Testing network with input ', arr, ' : ', layer.run_wo_thr(arr) )
        return


    def test_threshold(self):
        Functions.theta = 0.0
        layer = Layer(2, 1, threshold=Functions.theta, activation='threshold')

        x = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
            ])
        y = np.array([
            [1],
            [0],
            [1],
            [0]
            ])

        m_weights = np.full((layer.num_nodes, layer.num_in), 0)
        layer.init_weights(m_weights)

        layer.show("before training", x, y)  # before training
        # training
        layer.learn_w_thr(
            x,           # input matrix/vectors
            y,    # output target vector of "or" function
            0.1,            # learning rate
            )

        layer.show("after training", x, y)  # after training

        # Now test an input vector unknown to the net
        arr = np.array([-1, 0.2, 0.9])
        print('Testing network with input ', arr, ' : ', layer.run_wo_thr(arr))
        return



if __name__ == '__main__':
    # Main entry point for tests
    slp = SLP(2, 2)
    slp.test_sigmoid()