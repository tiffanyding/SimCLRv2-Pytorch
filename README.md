# SimCLRv2-Pytorch

## My procedure:
```
python download.py r152_3x_sk1
python convert.py r152_3x_sk1/model.ckpt-250228
python get_simclr_representations.py train r152_3x_sk1.pth
python train_linear_and_get_logits.py 
```

## Original README

Pretrained SimCLRv2 models in Pytorch

```python
python download.py r152_3x_sk1
python convert.py r152_3x_sk1/model.ckpt-250228 [--ema]
python verify.py r152_3x_sk1.pth
```

| Model | Pytorch | TF |
| :-------------: |:-------------:| :-----:|
| r50_1x_sk0 | 70.97 | 71.7 |
| r50_1x_sk1 | 73.79 | 74.6 |
| r152_3x_sk1 | 79.12 | 79.8 |
