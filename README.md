# Diformer

Diformer: A Dynamic Self-Differential Transformer for New Energy Power Autoregressive Prediction

New energy is gradually becoming a critical factor in the field of carbon capping and neutrality. In 2021, the amount of 257 GW of new renewable energy was established globally. Diformer achieves SOTA in eight new energy power prediction datasets.

## Diformer vs. Transformers & Linear

**1. Self differential attention**

A novel self-differential attention captures sequence trend.

**2. Trade-Loss**

A dynamic loss function is derived to balance the model’s exposure to periodicity and volatility in sequence modeling, enhancing the sequence prediction capability of the ontology model.

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from `./data`. The datasets password can be obtained by contacting the author. **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETT_script/Diformer.sh
bash ./scripts/ECL_script/Diformer.sh
bash ./scripts/PTT_script/Diformer.sh
```

## Baselines

We will keep adding series forecasting models to expand this repo:

- [x] Autoformer
- [x] Informer
- [x] Transformer
- [x] PatchTST
- [x] DLinear
- [x] HI
- [x] STID
- [x] FEDformer
- [x] LSTMa
- [x] DeepAR
- [ ] Reformer
- [ ] N-BEATS



