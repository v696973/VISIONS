# VISIONS
VISIONS (for VIsual-SemantIc navigatiOn in raNdom datasetS) is a tool for ranking and visual navigation in random data sets, consisting of images, text data, or their combinations.

The project is on it's early stage and is currently under heavy development.

## How it works
VISIONS utilizes several pre-trained deep learning models in order to construct embeddings for images and text data. These embeddings are then projected into a shared visual-semantic vector space. Projected vectors are then stored in an on-disk index and can be ranked by similarity to query vector or one of the items in the index.

## Install
VISIONS requires Python 3.6+ to work.
1. `git clone https://github.com/v696973/VISIONS.git && cd VISIONS`
2. (Optional, but highly recommended) Create virtualenv for the project and activate it
3. `pip install -r requirements.txt`

## Usage
See `python main.py --help` for help.
In order to query the dataset, VISIONS first needs to build an index for it. For now, only flat index is supported (flat index puts the embeddings into separate bucket files and then performs bruteforce nearest neighbour search on each bucket).

Use `python main.py build_index --data_dir <your dataset directory>` to build new index. 
For now, the project supports two basic dataset formats: flat directory with texts and/or images, and nested directory with mixed data. Flat directory data reader treats every separate file (either image or text file) as a separate entity. Mixed data reader requires dataset to be a directory with a separate subdirectory for every entity in the dataset. Each entity dir should have two subdirectories - `img` and `text` for images and text files respectively.

See `python main.py build_index --help`
for additional parameters.

Use `python main.py infer` mode allows to perform similarity queries on the dataset. See `python main.py infer --help` for more info.

At first run, VISIONS will download weights for pre-trained models, used for feature extraction. Weights for aligner models are included into this repo.

## Papers and repos used
* [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612) - aligner models were trained using MH loss described in the paper. https://github.com/fartashf/vsepp was used as a reference implementation of the loss function.
