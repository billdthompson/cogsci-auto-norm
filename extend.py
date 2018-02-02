# coding: utf-8

# Author:
# -----------
# Copyright (c) 2017 - Present Bill Thompson (biltho@mpi.nl) 
# 

import pandas as pd
import numpy as np

import click
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

N = 1000000 # skipgram max vocabulary size
D = 300 # skipgram vector dimension

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


@click.command()
@click.option('--language', '-l', default='nl')
@click.option('--norm', '-n', default='concreteness')
@click.option('--vectortransformfile', '-v', default='nl.txt')
@click.option('--coefficientfile', '-c', default='concreteness-norms-en-prediction-transform.coef')
def run(language, norm, vectortransformfile, coefficientfile):

    # find the semantic model for the new language
    logging.info('Preprocessing > Retrieving semantic model for {}'.format(language))
    skipgram = Skipgram('wiki.{0}.vec'.format(language))

    # extract its vectors and vocabulary
    logging.info('Preprocessing > Transforming {} vectors into English semantic space'.format(language))
    vectors = np.matmul(skipgram.vectors, np.loadtxt(vectortransformfile))

    # compute the predictions using dot prod of vectors and learned coef-vec
    logging.info("Processing > Transforming vectors into {0} predictions using {1}".format(norm, coefficientfile))
    predictions = pd.DataFrame({'word':skipgram.word, 'estimated-{}'.format(norm):np.dot(vectors, np.loadtxt(coefficientfile))})
    
    # save predictions out
    logging.info("Postprocessing > Saving predictions out to: {0}-estimates-{1}.csv".format(norm, language))
    predictions.to_csv('{0}-estimates-{1}.csv'.format(norm, language))

if __name__ == '__main__':
    run()








