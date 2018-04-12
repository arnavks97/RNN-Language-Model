#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:39:36 2018

@author: arnav
"""

import numpy as np
import theano
import theano.tensor as tt
import operator


class RNNLanguageModel:
    
    def __init__(self, vocab_size, hidden_size = 100, back_prop_through_time = 4):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.back_prop_through_time = back_prop_through_time
        
        #RNN parameters are randomly given initial values
        param1 = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_size, vocab_size))
        param2 = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (vocab_size, hidden_size))
        param3 = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))
        
        self.param1 = theano.shared(name = 'param1', value = param1.astype(theano.config.floatX))
        self.param2 = theano.shared(name = 'param2', value = param2.astype(theano.config.floatX))
        self.param3 = theano.shared(name = 'param3', value = param3.astype(theano.config.floatX))
        
        self.theano = {}
        self.__build__()
        
        
    def __build__(self):
        param1 = self.param1
        param2 = self.param2
        param3 = self.param3
        
        v1 = tt.ivector('v1')
        v2 = tt.ivector('v2')
        
        
        def fwd_prop(input_ss, hidden_ss_prev, param1, param2, param3):
            hidden_ss = tt.tanh(param1[:,input_ss] + param3.dot(hidden_ss_prev))
            output_ss = tt.nnet.softmax(param2.dot(hidden_ss))
            
            return [output_ss[0], hidden_ss]
        
        [output, hsaved], updates = theano.scan(fwd_prop, sequences = v1, outputs_info = [None, dict(initial = tt.zeros(self.hidden_size))], non_sequences = [param1, param2, param3], truncate_gradient = self.back_prop_through_time, strict = True)
        
        pred = tt.argmax(output, axis = 1)
        output_err = tt.sum(tt.nnet.categorical_crossentropy(output, v2))
        
        #Gradient for parameter 1
        deltaP1 = tt.grad(output_err, param1)
        #Gradient for parameter 2
        deltaP2 = tt.grad(output_err, param2)
        #Gradient for parameter 3
        deltaP3 = tt.grad(output_err, param3)
        
        self.forward_prop = theano.function([v1], output)
        self.prediction = theano.function([v1], pred)
        self.computational_err = theano.function([v1, v2], output_err)
        self.b_p_t_t = theano.function([v1, v2], [deltaP1, deltaP2, deltaP3])
        
        alpha = tt.scalar('alpha')
        self.grad_descent = theano.function([v1, v2, alpha], [], updates = [(self.param1, self.param1 - (alpha * deltaP1)), (self.param2, self.param2 - (alpha * deltaP2)), (self.param3, self.param3 - (alpha * deltaP3))])
        
        
    def complete_loss(self, V1, V2):
        return np.sum([self.computational_err(i, j) for i, j in zip(V1, V2)])
        
    def loss(self, V1, V2):
        num_training = np.sum((len(i) for i in V2))
        return self.complete_loss(V1, V2)/float(num_training)
        
        
def check_gradient(model, v1, v2, h = 0.001, err_limit = 0.01):
    model.back_prop_through_time = 1000
    grads = model.b_p_t_t(v1, v2)
    params = ['param1', 'param2', 'param3']
    
    for param_id, param_name in enumerate(params):
        paramT = operator.attrgetter(param_name)(model)
        param = paramT.get_value()
        print "Checking gradient for param %s of sz %d" % (param_name, np.prod(param.shape))
        
        iterator = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        
        while not iterator.finished:
            i = iterator.multi_index
            val_real = param[i]
            
            # Gradient estimation by (fun(x+h) - fun(x-h))/(2h)
            param[i] = val_real + h
            paramT.set_value(param)
            positive_grad = model.complete_loss([v1], [v2])
            param[i] = val_real - h
            paramT.set_value(param)
            negative_grad = model.complete_loss([v1], [v2])
            
            grad_est = (positive_grad - negative_grad)/(2*h)
            
            param[i] = val_real
            paramT.set_value(param)
            
            grad_bptt = grads[param_id][i]
            
            # Relative err = (|(+veGrad) - (-veGrad)|/(|+veGrad| + |-veGrad|))
            rel_err = np.abs(grad_bptt - grad_est)/(np.abs(grad_bptt) + np.abs(grad_est))
            
            if rel_err > err_limit:
                print "Gradient Error: Parameter's Name = %s, i = %s" % (param_name, i)
                print "Value of loss x+h: %f" % positive_grad
                print "Value of loss x-h: %f" % negative_grad
                print "Value of Estimated Gradient: %f" % grad_est
                print "Value of Backpropagation Gradient: %f" % grad_bptt
                print "Value of Relative Error: %f" % rel_err
                return
                
            iterator.iternext()
            print "Gradient check for the given parameter %s passed." % (param_name)