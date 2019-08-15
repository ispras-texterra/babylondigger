# BabylonDigger
Toolkit for text segmentation, part-of-speech tagging, lemmatization and dependency parsing.

**BabylonDigger** can be used for training neural network-based morphological analyzers for [UD](https://universaldependencies.org/) languages.
The toolkit allows building custom models, providing a wide range of options for features, sentence-level context retrieval and training optimization.

For features, apart from standard word embeddings and characters it is also possible to use byte-pair encoding segments or Morfessor-generated morphemes with pre-trained embeddings. 
The feature extraction options include averaging, CNNs, biLSTM, and self-attention over subword embeddings. 

For retrieving sentence-level contextual information, the toolkit provides architectures based on biLSTM and dilated CNNs.
Optionally, highway layers and skip-connections can be added to improve the information flow in the network.

## Training & Evaluation

To train and test a model, use script `evaluation/evaluation.py`, which has the following arguments:
```
--settings       path to JSON file the experiment configuration
--method         method to be tested
--hyperparams    path to JSON file with the neural network specification and hyperparameters
--delete_after   an optional argument to indicate whether to delete models after the experiment or not
```
A more detailed description of the arguments and available options are described below.

### Experiment Configuration (--settings)

This file defines the experiment's setup such as train, test, development files and random seeds (a sample configuration file is provided in `resources/settings.json`).

The values in `random_seed` are used for the random initialization of models during training, and the final evaluation results are the average for specified seeds.
`train_set`, `dev_set`, `test_set` are used to specify the path to CoNLL-U dataset files, 
and the latter also have an optional `results_path` parameter allowing to save the predictions. 

`fields` in `cmp_metric` are used to compare the model's performance between epochs and determine the best. 
The sum of specified metric's values for given fields is used during comparison. 

`metric_printer` allows to specify for which fields print the performance scores during training and evaluation.

### Method (--method)

- `neural`: trains and evaluates a model using given train, development and test datasets.
- `neural_pretrained`: evaluates the pretrained models on given development and test datasets.

### Neural Network Specification and Hyperparameters (--hyperparams)

The neural network's input features, internal layers, learning tasks and setup can be defined in a JSON file 
(a sample is provided in `resources/hyperparameters.json`).

#### Shared layers

The input layers of the neural network are shared between different tasks. 
These shared layers include the feature extractors, as well as sentence-level layers that leverage context information.
The output of the shared part is passed to task-specific layers, parameters of which are not shared anymore. 

The architecture and hyperparameters are specified in `shared_part` entry. The following architectures are available for shared layers:

- `WE_charCNN`: builds the specified feature set and returns their concatenation.
- `BiLSTM`: builds the specified feature set, applies a multilayer biLSTM to their concatenation and returns the concatenation of final biLSTM layer's final states.
- `BiLSTM_CNN`: similar to `BiLSTM` option, but the output is additionaly processed through sentence-level CNN layers.
- `Dilated_CNN`: builds the specified feature set and applies sentence-level dilated CNN layers to their concatenation.

`sparse_input_prob` can be used to randomly replace input features with the UNKNOWN token with the specified probability.
This may help with overfitting.

`skip_connection` is another optional parameter that allows to add a skip-connection from the first shared layer to the last. 
The connection type should be `concat` or `sum` (for concatenating and summing the values respectively).

#### Feature set

The input features of the model are specified in `features_config` of `shared_part`.
With the toolkit, it is possible to build models that employ both word-level and subword-level features. 

**Word-level features**: pretrained vectors such as GloVe, fastText, word2vec can be used for initializing word embeddings.
Parameter `trainable` controls whether the embeddings are updated during training or not.
If `lowercase` is `true`, all words are lowercased before retrieving the embedding.

The toolkit also allows to use embeddings with random initialization, which are then updated during training.
`min_frequency` is used to filter rare words out and train embeddings only for frequent words.

- pretrained word embeddings:
```
"word_embedding": {
  "we_dense_size": 100,
  "path": "<path to the word embeddings file>",
  "trainable": false,
  "lowercase": true
}
```
- trained word embeddings:
```
"word_trained_embedding": {
  "trainable": true,
  "size": 50,
  "min_frequency": 5,
  "lowercase": true
}
```

**Subword features**: The toolkit allows to build features based on characters, word's byte-pair encoding (BPE) segments, and Morfessor-generated morphemes.
The subwords are embedded into dense representations (embeddings). 
It is possible to provide pretrained embeddings (similar to word vectors) for subwords through `embedding_path` parameter.
However, this is optional and if not specified, the embeddings are initialized randomly.

The embeddings of each input token's subwords are aggregated through a function into a single feature vector.
The function is set through `aggregator` parameter, and the following options are available:

- `average`: returns the average of subwords' embeddings.
- `cnn`: applies consecutive CNN layers over subwords' embeddings and returns the result of final layer's max pooling. 
Kernel size, count and dilation are specified in `aggregator_config`:
```
  ...
  "aggregator": "cnn",
  "aggregator_config": {
    "cnn_kernels": [
      { "size": 3, "count": 512, "dilation": 1},
      { "size": 3, "count": 124, "dilation": 2},
      { "size": 3, "count": 64, "dilation": 4}
    ]
  },
  ...
```
- `rnn`: runs bidirectional LSTM layers over subwords' embeddings and returns the sum of final states. 
The number of LSTM units and layers is specified in  `aggregator_config`:
```
  ...
  "aggregator": "rnn",
  "aggregator_config": {
    "lstm_cell_size": 50,
    "bilstm_layer_count": 1
  },
  ...
```
- `self-attention`: returns the result of self-attention on subwords' embeddings.

Below are given sample feature configurations for each available subword segmentation:

- character-based features:
```
"word_characters": {
  "segmenter": "char",
  "embed_size": 50,
  "min_frequency": 1,
  "trainable": true,
  "aggregator": "cnn",
  "aggregator_config": {
    "cnn_kernels": [
      { "size": 3, "count": 512, "dilation": 1},
      { "size": 3, "count": 124, "dilation": 2},
      { "size": 3, "count": 64, "dilation": 4}
    ]
  },
  "highway_layer": true
}
```

- BPE-based features:
```
"bpe_subwords": {
  "segmenter": "bpe",
  "segmenter_config": {
    "language": "hy",
    "vocab_size": 50000
  },
  "min_frequency": 1,
  "embedding_path": "<bpe subwords embeddings path>",
  "trainable": true,
  "aggregator": "rnn",
  "aggregator_config": {
    "lstm_cell_size": 50,
    "bilstm_layer_count": 1
  },
  "highway_layer": true
}
```
BPE segmentation is performed using the models from [BPEmb](https://nlp.h-its.org/bpemb/) project. 
`vocab_size` and `language` in `segmenter_config` determine which BPEmb model to use.

- Morfessor-based features:
```
"Morfessor_subwords": {
  "segmenter": "morfessor",
  "segmenter_config": {
    "path": "<morfessor path>"
  },
  "embedding_path": "<morfessor subwords embeddings path>",
  "embed_size": 50,
  "min_frequency": 1,
  "trainable": true,
  "aggregator": "average",
  "highway_layer": true
}
```
Optionally, a highway layer can be added to subword features by setting `highway_layer` parameter to `true`.

#### Tasks

The toolkit allows to train the morphological analyzer and the lemmatizer jointly as well as separately. 
The tasks are designated as `pos` and `lemma` respectively. 
To train them jointly, include both values in `tasks` parameter, otherwise specify only one of them. 

The included tasks' layers are configured in an entry with that task's name. 
In the entry, `builder` defines which architecture to use for the task and in `config` its hyperparameters are specified.

**pos**: for the task of predicting morphological tags, the toolkit uses a dense layer followed by a softmax layer for each target tag (`pos_dense` builder). 
There are several options for target tags, determined by the value of parameter `splitting`. 
The parameters accepts the following values:

- `pos_joint`: concatenates UPOS, XPOS, FEATS tags into a joint target.
- `pos_XF`: concatenates XPOS and FEATS tags into a joint target (UPOS is predicted separately).
- `pos_UXF`: predicts UPOS, XPOS, FEATS tags as separate targets.
- `pos_gran`: splits the value in FEATS into separate tags and along with UPOS and XPOS predicts as independent targets.

**lemma**: there are various lemmatizer architectures available for use (parameter `builder`):

- `lemma_suffix`: the suffix transformation from form to its lemma is predicted, then the lemma is constructed using the predicted transformation.
- `lemma_combo`: CNN-based subword transducer similar to the lemmatizer from [COMBO](https://github.com/360er0/COMBO) parser.
- `lemma_dense`: softmax layer that classifies among lemmas seen in training data.

During joint training, the sum of these tasks' losses is used as the loss of the network. 
Parameter `loss_weights` in each task's `config` allows to specify its weight for weighted sum of tasks' losses.

#### Optimizer

The toolkit allows the use of following optimizers:

- `GD`: gradient descent.
- `Adam`: Adam optimizer.
- `RMSProp`: RMSProp optimizer.

`lr_decrease_factor` can be used to drop the learning rate by the specified factor (default is 2) when the model stagnates during training.
Use value `1` if you don't want to use this feature.

`lr_decay` applies a time-based decay of the learning rate after each epoch. If `0.0`, the learning rate does not decay.
