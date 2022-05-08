# Phishing-Scam-Detection-on-Ethereum-with-GCN

Aiming at the phishing scams of Ethereum, this project built a phishing fraud node detection model based on feature engineering and Graph Convolutional Neural Network. First of all, from the Ethereum server and authoritative websites I crawled the Ethereum transaction data, from which I created a Ethereum transaction network; According to the transaction information of nodes and their first-order adjacent nodes, I created a total of 120 features including 20 first-order features and 100 two-class associative fusion features. Finally, the Graph Convolutional Neural Network model was applied to construct the phishing fraud classifier. ( The GCN model is based on the implementation of )

Experimental results show that the model has a good effect on the identification of phishing nodes in Ethereum, and provides a new idea for network fraud detection in Ethereum. The features extracted by this model can reflect the nature of phishing fraud nodes, and at the same time solve the problem of data imbalance in existing methods.

## References
[1] Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

[2]
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)
