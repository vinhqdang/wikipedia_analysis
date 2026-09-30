# Legacy code (2016-2017)

The original code from the first version of this repository: R scripts (`analysis/`, `analysis_2/`,
`vandalism/`) and Keras/tflearn LSTM, GRU, CNN and doc2vec models (`lang_model/`).
It targets R 3.2.3, Python 2 and TensorFlow 1.x and is kept for reference only.

The large data files (article text, ORES feature tables, contribution files) were removed from the
working tree. They are still in the git tag `legacy-2017`:

```sh
git checkout legacy-2017
```

The current pipeline is in the repository root.
