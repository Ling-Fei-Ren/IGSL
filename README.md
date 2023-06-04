# PC-GNN

This is the author implementation of "[Improving Fraud Detection via Imbalanced Graph Structure Learning]" .


## Requirements

```
argparse          1.1.0
networkx          1.11
numpy             1.16.4
scikit_learn      0.21rc2
scipy             1.2.1
torch             1.4.0
```

## Dataset
YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).


Kindly note that there may be two versions of node features for YelpChi. The old version has a dimension of 100 and the new version is 32. In our paper, the results are reported based on the old features.

## Usage

```sh
python main.py --config ./config/igsl_yelpchi.yml
```
python main.py --config ./config/igsl_amazon.yml

```
