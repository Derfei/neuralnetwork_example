'''
author: longxin
date: 2019-11-12
description: the basic example of neural networks
changeversion: 0.0
changedescripiton:

'''

'''
the example of full connected layer
'''
class fullconnected_layer:

    def __init__(self, x_shape, neural_num):
        '''
        init function
        :param x_shape: the shape of init
        :param neural_num: the count of neural
        '''
        import numpy as np

        self.W = np.random.random(size=[neural_num] + list(x_shape))
        self.bias = np.random.random()


    def computate(self,x):
        import numpy as np
        return np.dot(self.W, x) + self.bias

'''
the example of convenlutaion network
'''
class convenlution_layer:

    def __init__(self):
        pass

    def computer(self, x):
        pass