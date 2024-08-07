# quantile pooling

This repository contains the CUDA & PyTorch implementation of the quantile pooling layer as described in the paper "Stacking Deep Set Networks and Pooling by Quantiles" by [Chen et al.](https://openreview.net/forum?id=Lgq1E92h1U).

```
@inproceedings{chenstacking,
    title={Stacking Deep Set Networks and Pooling by Quantiles},
    author={Chen, Zhuojun and Zhu, Xinghua and Su, Dongzhe and Chuang, Justin CI},
    booktitle={Forty-first International Conference on Machine Learning}
}
```


## Installation
### Requirements
- Python 3.6+
- PyTorch 1.0+

To install the package, run the following command:

```bash
cd quantile_pooling
python setup.py install
```

To test the installation:
    
```bash
python test.py
```

## Usage
Please refer to the script `quantile_pooling/quantile_pooling/quantile_pooling.py`,
or follow the example in `test.ipynb`. 

