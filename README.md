# NAO
Nonlocal Attention Operator

![NIPS architecture.](https://github.com/fishmoon1234/NAO/blob/main/NIPS.png)

This repository houses the code for our papers in attention-mechanism-based neural operators, published on NeurIPS 2024 and ICML 2025:
- [NAO: Nonlocal Attention Operator: Materializing Hidden Knowledge Towards Interpretable Physics Discovery](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce5b4f79f4752b7f8e983a80ebcd9c7a-Abstract-Conference.html)
- [NIPS: Neural Interpretable PDEs: Harmonizing Fourier Insights with Attention for Scalable and Interpretable Physics Discovery]


**Highlights**: 

1. Nonlocal Attention Operator features a incorporation of attention mechanism into neural operator architectures, which simultaneously performs physics modeling (forward PDE) and mechanism discovery (inverse PDE).

2. Neural Interpretable PDEs (NIPS) are an enhanced version of NAO, with learnable Fourier kernel and linear attention. It provides a substantial leap in scalable, interpretable, and efficient physics learning.

3. Once trained, NAO and NIPS are zero-shot generalizable to unseen physical systems in predicting system responses and discovering hidden PDE parameters.

## Requirements
- [PyTorch](https://pytorch.org/)


## Running experiments

**NAO Example 1**:

To run the synthetic dataset (example 1 with sine kernels only) in the NAO paper
```
python3 Attention_synthetic_int.py 
```
Pretrained models with 2 layers and feature dimension=302 are provided, corresponding to Table 2 in the paper:
```
best_model_dk[10/20/40]_sinonly_synthetic.ckpt
```

To run the synthetic dataset with more kernels (example 1 with sine+cos+poly kernels), just uncomment lines 163-185, 212-238, then run:
```
python3 Attention_synthetic_int.py 
```
Pretrained models with 2 layers and feature dimension=302 are provided, corresponding to Table 2 in the paper:
```
best_model_dk[10/20/40]_diverse_synthetic.ckpt
```


**NAO Example 2**:

To run the solution operator dataset (example 2) with linear (g to p) setting in the NAO paper
```
python3 Attention_darcy_gtou_int.py
```
Pretrained models with 9000 training samples, 2 layers and feature dimension=50 are provided, corresponding to Table 2 in the paper:
```
best_model_dk40_darcy_gtou.ckpt
```
A more accurate pretrained model with deeper (4) layers:
```
best_model_dk40_darcy_gtou_deeper.ckpt
```

To run the solution operator dataset (example 2) with nonlinear (b to p) setting in the NAO paper
```
python3 Attention_darcy_chitou_int.py
```


**NIPS 2D Darcy example**:
```
python3 NIPS_Darcy.py
python3 LinearNAO_Darcy.py
```
**NIPS Mechanical MNIST example**:
```
python3 NIPS_MMNIST.py
```
## NIPS Datasets
We provide the Darcy and MMNIST datasets that are used in the paper.

- [Darcy and MMNIST datasets](https://drive.google.com/drive/folders/1-HA5uPMBHEH96sRcdzKaF7dyn8KQv8kG?usp=sharing)

## Citation

If you find our models useful, please consider citing our papers:

```
@inproceedings{liu2025nips,
  title={Neural Interpretable PDEs: Harmonizing Fourier Insights with Attention for Scalable and Interpretable Physics Discovery},
  author={Liu, Ning and Yu, Yue},
  booktitle={Proceedings of the 42th International Conference on Machine Learning (ICML 2025)}
}
@inproceedings{yu2024nonlocal,
  title={Nonlocal Attention Operator: Materializing Hidden Knowledge Towards Interpretable Physics Discovery},
  author={Yu, Yue and Liu, Ning and Lu, Fei and Gao, Tian and Jafarzadeh, Siavash and Silling, Stewart},
  booktitle={Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)}
}
```
