## Concreteness Estimates in 50 Languages

**Author**: Bill Thompson (biltho@mpi.nl)


#### Summary
This repository contains concreteness estimates in 50 languages, and code to produce further estimates.  

---

#### Workflow

Here's how you would compute concreteness estimates in Dutch from a set of experimental lexical norms of concreteness.

1) Clone this repo.

2) Download the English and Dutch [Wikipedia-trained Skipgram semantic models](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) released in 2017 by [Facebook Artificial Intelligence Research](https://research.fb.com/category/facebook-ai-research-fair/). Be sure to download the .vec versions of these models (i.e. [wiki.en.vec](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec), [wiki.nl.vec](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.nl.vec)), and move the files into this directory. 

3) Ensure this directory contains a csv file with the experimental norms on which you want to train a model. This file should have a column named ```word``` and a column named ```concreteness``` (or whatever norm you are training on). This repository already includes the file ```norms.csv``` which contains the [Brysbaert Concreteness norms for English](https://link.springer.com/article/10.3758%2Fs13428-013-0403-5).

4) Train a simple linear model to predict concreteness from semantic vectors by running:

```python distill.py -l en -n concreteness -f norms.csv```

This will result in a new dataset of estimated concreteness norms in English, and a vector of estimated coefficients in the linear regression (*here's one i made ealier*: ```concreteness-norms-en-prediction-transform.coef```).  

5) Transform the Dutch semantic model into English semantic space using a vector-alignment transform (such as those released by [Babylon](https://www.babylonhealth.com/) and availible [here for 78 languages](https://github.com/Babylonpartners/fastText_multilingual)). This repository already contains the file ```nl.txt```, which is the Babylon-released transform for Dutch. Then apply the inferred regression coefficients to this transformed semantic model. All this is achieved by running:

```python extend.py -l nl -n concreteness -v nl.txt -c concreteness-norms-en-prediction-transform.coef```

This will produce a new file ```concreteness-estimates-nl.csv``` containing estimates of concreteness for the most frequent ```N``` terms in the Skipgram vocabulary for Dutch (```N = 100000``` by default; change this in ```extend.py```).
