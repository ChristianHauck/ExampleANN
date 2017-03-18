#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

import numpy as np
import struct


class Utils:

    mnist_train_data   = "train-images.idx3-ubyte"
    mnist_train_labels = "train-labels.idx1-ubyte"
    mnist_test_data    = "t10k-images.idx3-ubyte"
    mnist_test_labels  = "t10k-labels.idx1-ubyte"


    @classmethod
    def load_mnist_data(cls, file_name, max_items=-1, only_subset=False):
        f = open('./data/' + file_name, 'rb')  # do not forget the 'b' for binary read!
        magic = f.read(4)  # ignore the "magic" number...
        num_items = struct.unpack('>i', f.read(4))[0]
        if max_items >= 0:
            num_items = max_items
        rows = struct.unpack('>i', f.read(4))[0]
        columns = struct.unpack('>i', f.read(4))[0]

        img_size = rows * columns
        #
        data = np.fromfile(f, dtype=np.uint8, count=num_items*img_size)
        f.close()
        assert num_items*img_size == data.nbytes == data.size
        #
        data = data.reshape((num_items, img_size))
        data = data / 255.0  # keep unaugmented assignment because I do not trust numpy so far ;)
        #
        images = []
        for d in data:
            img_vec = np.zeros((img_size + 1,), dtype=float)    # +1 for the mandatory -1
            img_vec[0] = -1.0
            # Option to make the data much smaller for testing purposes
            if only_subset:
                img_vec[1:] = d[14]  # take only row 14 - it should hold some pixel set
            else:
                img_vec[1:] = d
            images.append(img_vec)
        #
        return [num_items, rows, columns, images, data]


    @classmethod
    def load_mnist_labels(cls, file_name, max_items=-1):
        f = open('./data/' + file_name, 'rb')
        magic = f.read(4)  # ignore
        num_items = struct.unpack('>i', f.read(4))[0]
        if max_items >= 0:
            num_items = max_items
        data = np.fromfile(f, dtype=np.uint8, count=num_items)
        f.close()
        assert num_items == data.nbytes
        #
        labels = []
        for d in data:
            lbl_vec = np.zeros((10,), dtype=float)
            lbl_vec[d] = 1.0
            labels.append(lbl_vec)
        #
        return [num_items, labels, data]

