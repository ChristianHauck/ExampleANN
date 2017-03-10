#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

from mlp import MLP
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import time
from thread import start_new_thread


class Program:

    def __init__(self):
        self.m_x = []
        self.m_y = []


    def set_input(self):
        self.m_x.append(np.array([-1, 0, 0]))
        self.m_x.append(np.array([-1, 0, 1]))
        self.m_x.append(np.array([-1, 1, 0]))
        self.m_x.append(np.array([-1, 1, 1]))
        
        self.m_y.append(np.array([0.1, 0.2]))
        self.m_y.append(np.array([0.3, 0.4]))
        self.m_y.append(np.array([0.6, 0.6]))
        self.m_y.append(np.array([0.9, 0.8]))
        
        return


    def test(self, mlp, learn_rate, num_loops):
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


    def run(self, mlp, m_x, m_y, learn_rate, num_loops):
        # training
        mlp.learn(
            m_x,        # input matrix/vectors
            m_y,        # output target vector of "or" function
            learn_rate, # learning rate
            num_loops   # number of loops
            )
        # Now test an input vector unknown to the net
        result = mlp.run(m_x[0])
        # print "Testing network with trained input. Result:"
        # print result
        #
        return


    def view_image(self, image_data):
        image_data = image_data.reshape((28, 28))
        fig, ax = plt.subplots()
        ax.imshow(image_data, cmap='gray')
        plt.show()



if __name__ == '__main__':

    prog = Program()

    ###--- Load images and labels from MNIST data set ---###
    max_items = 30
    num_images, rows, columns, images, raw_images = \
        Utils.load_mnist_data(Utils.mnist_train_data, max_items=max_items)
    num_labels, labels, lbl_vals = \
        Utils.load_mnist_labels(Utils.mnist_train_labels, max_items=max_items)

    ###--- Print one data set ---###
    print "num_images=", num_images, " rows=", rows, " columns=", columns
    # print images[0]
    print "num_labels=", num_labels
    print labels[0], " = ", lbl_vals[0]
    # for i in range(max_items):
    #     prog.view_image(raw_images[i])

    ###--- Train and recognize one data set ---###
    prog.set_input()
    input_dim = prog.m_x[0].size
    print "input_dim=", input_dim
    mlp = MLP([input_dim, 5, 2])
    mlp.init_layers()
    mlp.set_activation("sigmoid")
    prog.run(mlp, prog.m_x, prog.m_y, 5, 1000)
    mlp.show("Trained Network", prog.m_x, prog.m_y)

    #'''
    ###--- Train and recognize one data set ---###
    input_dim = images[0].size
    print "input_dim=", input_dim
    mlp = MLP([input_dim, 25, 10])
    mlp.init_layers()
    mlp.set_activation("sigmoid")
    prog.run(mlp, images, labels, 0.2, 5000)
    mlp.show("Trained Network", images, labels)
    #'''