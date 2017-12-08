# Abstractive summarization with OpenNMT-py

This is a [Pytorch](https://github.com/pytorch/pytorch)
implementation of Abstractive summarization methods on top
of [OpenNMT](https://github.com/OpenNMT/OpenNMT). It features vanilla attention seq-to-seq LSTMs,
[pointer-generator networks (See 2017)](https://arxiv.org/abs/1704.04368) ("copy attention"),
as well as [transformer networks  (Vaswani 2017)](https://arxiv.org/pdf/1706.03762.pdf)  ("attention is all you need")
as well as instructions to run the networks on both the Gigaword and the CNN/Dayly Mail datasets.


Table of Contents
=================

  * [Requirements](#requirements)
  * [Implemented models](#implemented models)
  * [Quickstart](#quickstart)
  * [Results](#results)
  * [Pretrained models](#models)

## Requirements

```bash
pip install -r requirements.txt
```

## Implemented models

The following models are implemented:

- Vanilla attention LSTM encoder-decoder
- Pointer-generator networks: ["Get To The Point: Summarization with Pointer-Generator Networks",
  See et al., 2017](http://arxiv.org/abs/1704.04368)
- Transformer networks: ["Attention is all you need", Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762)

## Quickstart

### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -share_vocab -dynamic_dict -src_vocab_size 50000
```

The data can be either Gigaword or the CNN/Daily Mail dataset.
* Mention details of how those were preprocessed and how to retrieve them *

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training.

After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

The basic command would be:

```bash
python train.py -data data/demo -save_model demo_model -share_embeddings
```

The main relevant parameters to be changed for summarization are:

* pointer\_gen to enable Pointer Generator
* to enable Transformer networks
* word\_vec\_size (128 has given good results)
* rnn\_size (256 or 512 work well in practice)
* encoder\_type (brnn works best on most models)
* layers (1 or 2)
* gpuid (0 for the first gpu, -1 if on cpu)

The parameters for our trained models are described below

### Step 3: Summarize

```bash
python translate.py -model demo-model_epochX_PPL.pt -src data/src-test.txt -o output_pred.txt -replace_unk -beam_size 10
-dynamic_dict -share_vocab
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

### Step 4: Evaluate with ROUGE

Perplexity and accuracy are not the main evaluation metrics for summarization. Rather, the field uses
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))
To evaluate for rouge, we use [files2rouge](https://github.com/pltrdy/files2rouge), which itself uses
[pythonrouge](https://github.com/tagucci/pythonrouge).

Installation instructions:

```bash
pip install git+https://github.com/tagucci/pythonrouge.git
git clone https://github.com/pltrdy/files2rouge.git
cd files2rouge
python setup_rouge.py
python setup.py install
```

To run evaluation, simply run:
```bash
files2rouge summaries.txt references.txt
```
## Pretrained embeddings (e.g. GloVe)

Go to tutorial: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://forum.opennmt.net/t/how-to-use-glove-pre-trained-embeddings-in-opennmt-py/1011)


## Results

Table to be included

## Pretrained models

Pretrained modells to be included
