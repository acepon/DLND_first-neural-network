#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

class GridSearch():

    def __init__(self, model, metric, tf, tt, vf, vt):

        self.model = model
        self.metric = metric
        self._tf = tf
        self._tt = tt
        self._vf = vf
        self._vt = vt
    
    def run(self, parameter_ranges):
        train_features = self._tf
        train_targets = self._tt
        val_features = self._vf
        val_targets = self._vt

        self._keep = dict()
        self._new_params = {k:v[0] for k,v in parameter_ranges.items()}

        for k, v in parameter_ranges.items():
            val_records = dict()
            print('\nEvaluating {} item'.format(k))
            for value in v:
                self._new_params[k] = value
                print('\n* {}'.format(str(self._new_params)))

                # train process
                network = self.model(train_features.shape[1],
                                    self._new_params.get('hidden_nodes'),
                                    1,
                                    self._new_params.get('learning_rate'),
                                    self._new_params.get('dropout'))
                losses = {'train':[], 'validation':[]}

                for ii in range(self._new_params.get('iters')):
                    # Go through a random batch of 128 records from the training data set
                    batch = np.random.choice(train_features.index, size=128)
                    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                                            
                    network.train(X, y)
                    
                    # Printing out the training progress
                    train_loss = self.metric(network.run(train_features).T, train_targets['cnt'].values)
                    val_loss = self.metric(network.run(val_features).T, val_targets['cnt'].values)
                    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(self._new_params.get('iters'))) \
                                    + "% ... Training loss: " + str(train_loss)\
                                    + " ... Validation loss: " + str(val_loss))
                    sys.stdout.flush()
                    
                    losses['train'].append(train_loss)
                    losses['validation'].append(val_loss)

                val_records[value] = losses.get('validation')[-1]
                self._keep[str(self._new_params)] = losses

            self._new_params[k] = min(val_records, key=val_records.get)


    @property
    def all_records(self):
        return self._keep

    @property
    def final_parameters(self):
        return self._new_params

        