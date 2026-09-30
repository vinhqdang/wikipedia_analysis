# RNN-LSTM for assessing quality of Wikipedia articles.

This repo contains our implementation of using bidirectional LSTM for automatically assessing the quality of Wikipedia articles.

The solution is validated against English, French and Russian Wikipedia datasets.

One of the most important advantages of this approach compare to handed-feature machine learning approach is that it can be applied to any language without prior knowledge about this language.

## Motivation

Feature engineering is really hard. It requires a lot of expertise from the designers. Given a new problem, you will need (a lot of) time to understand the problem before designing a feature set. 

RNN-LSTM could learn directly from the dataset, therefore we can apply it to any Wikipedia language. For instance, I don't know French nor Russian, but it does not prevent me to build a model to predict quality categories of these datasets.

## Implementation

The code is written using [keras](keras.io) with [tensorflow](tensorflow.org) as the backend. You will need a huge amount of memory, and would be better if you are equipped with powerful GPU(s).

### Update

(This information comes to me after my implementation)

It seems that bidirectional LSTM has became a *de facto* for NLP in 2017, as claimed by [Christopher Manning](https://nlp.stanford.edu/manning/).

He said in a recent [presentation](https://simons.berkeley.edu/sites/default/files/docs/6449/christophermanning.pdf) at Simons Institute, Berkeley that:

> To a first approximation,
the de facto consensus in NLP in 2017 is
that no matter what the task,
you throw a BiLSTM at it, with
attention if you need information flow

I wouldn't say biLSTM will solve all the problems. But I am in the main trend.



