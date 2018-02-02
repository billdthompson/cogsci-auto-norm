# coding: utf-8

# Author:
# -----------
# Copyright (c) 2017 - Present Bill Thompson (biltho@mpi.nl) 
# 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler as Scaler

import click
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

N = 1000000 # skipgram max vocabulary size
D = 300 # skipgram vector dimension

# lightweight wrapper for interfacing with .vec Skip-gram files 
class Skipgram:
    def __init__(self, modelpath=''):
        with open(modelpath, 'r') as f:
            # skip header
            next(f)
            
            self.vectors = np.zeros((N, D))
            self.word = np.empty(N, dtype = object)
            for i, line in enumerate(f):
        
                if i >= N: break

                rowentries = line.rstrip('\n').split(' ')
                self.word[i] = rowentries[0]
                self.vectors[i] = rowentries[1:D + 1]

            self.vectors = self.vectors[:i]
            self.word = self.word[:i]



@click.command()
@click.option('--filename', '-f', default='norms.csv')
@click.option('--language', '-l', default='en')
@click.option('--norm', '-n', default='concreteness')
def run(filename, language, norm):
    
    ### Data Preprocessing --->
    logging.info('Preprocessing > Reading data from {0}'.format(filename))
    data = pd.read_csv(filename)[['word', norm]].dropna().drop_duplicates(subset = 'word').copy()

    # format norms
    logging.info('Preprocessing > Centering {0} norms to zero-mean'.format(norm))
    norm_mean, norm_min, norm_max = data[norm].mean(), data[norm].min(), data[norm].max()
    data[norm] -= norm_mean
    data['word'] = data.word.str.lower()

    # obtain semantics
    logging.info('Preprocessing > Retrieving semantic model for {}'.format(language))
    skipgram = Skipgram('wiki.{0}.vec'.format(language))
    
    # merge norms and semantics
    logging.info('Preprocessing > Merging Skipgram and experimental vocabularies')
    skipgramdata = pd.DataFrame(dict(word = skipgram.word))
    covered = skipgramdata.word.isin(data.word)
    skipgramdata = skipgramdata.merge(data, on = 'word', how = 'left')
    
    # silo training material
    logging.info('Preprocessing > Establishing resource overlap')
    training_vectors = skipgram.vectors[covered]
    logging.info('Preprocessing > Skipgram covers {0} (of {1}) experimentally normed words'.format(training_vectors.shape[0], data.shape[0]))

    # learn the regression
    logging.info('Regression > Regressing Experimental on raw semantic vectors')
    X, y = training_vectors, skipgramdata[covered][norm].values
    lr = LR()
    lr.fit(X, y)

    # save out the coefficients for prediction
    predictiontransformfile = '{0}-norms-{1}-prediction-transform.coef'.format(norm, language)
    logging.info('Saving coefs for prediction our to: {}'.format(predictiontransformfile))
    np.savetxt(predictiontransformfile, lr.coef_)
    
    # predict new values (and re-center at empirical mean)
    logging.info('Regression > Predicting full vector vocabulary')
    predictions = lr.predict(skipgram.vectors) + norm_mean
    
    # scale predictions back into legal range (empirically observed range) 
    skipgramdata['estimated-{}'.format(norm)] = Scaler((norm_min, norm_max)).fit_transform(predictions.reshape(-1, 1))
    
    # re-center the original norms
    skipgramdata[norm] += norm_mean

    # click save
    logging.info('Regression > Predictions correlate with experimental observation at: {}'.format(skipgramdata.corr()[norm]['estimated-{}'.format(norm)]))
    new_filename = "{0}-estimates-{1}.csv".format(norm, language)
    skipgramdata.to_csv(new_filename, index = False)


if __name__ == '__main__':
    run()








