This folder consists of main base jupyter notebook files out of numerous experiments conducted for training and finetuning processes.

# Code

It is advisory to create your own virtual environment in python and all the dependencies can be installed from `requirements.txt`.

However, PyTorch installation is required for Transformer models, and you may be required to install a specific version based on your CUDA version. More info on installing PyTorch can be found [here](https://pytorch.org/get-started/locally).

_**Note:** Transformers and RNN codes should be run on a dedicated GPU._

## Subfolder contents

### [Data Collection](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/data-collection)

* Nepali news scraping Script
* Exploratory Data Analysis

### [Preprocessing](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/preprocessing)

* Preprocessing Script
* Train-Valid-Test split with varying dataset sizes

### [Transformer models](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/transformer-models)

* Intuitive evaluation of models using Word Masking
* Tokenization examples with pretrained tokenizers
* Finetuning using Trainer API
* Weights and Biases (wandb) logging for Transformers

### [RNN model](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/rnn-models)

* BiLSTM - Model architecture
* Training, Validation and Testing
* Weights and Biases (wandb) logging for Tensorflow

### [Machine Learning models](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/machine-learning-models)

* Nested Cross Validation Procedure using Grid Search
* Training with optimal hyperparameters and Testing

### [Results](https://github.com/Suyogyart/nepali-transformers-evaluation/tree/main/code/results)

* Comparison of models with `16-class` and `20-class` datasets with 1500 rows per class.
* Comparison of models with `20-class` dataset with 1500, 500, 250 and 50 rows per class (varying data sizes)
