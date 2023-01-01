# Filling Time series with DCGAN

## Train

### 1. Timeseries filling

```bash
python train.py \
    --datapath ../data/Binance_ETHUSDT_1h.csv \
    --mask_where random \
    --mask_rate 0.4 \
    --test_size 0.3 \
    --seq_len 64 \
    --skip 1 \
    --batch_size 128 \
    --num_iterations 10000 \
    --num_epochs 200 \
    --input_dim 1 \
    --noise_dim 10 \
    --ngf 64 \
    --ndf 64 \
    --learning_rate 2e-4 \
    --print_every 200 \
    --checkpoint_every 10000
```

### 2. Timeseries prediction

```bash
python train.py \
    --datapath ../data/Binance_ETHUSDT_1h.csv \
    --mask_where end \
    --mask_rate 0.4 \
    --test_size 0.3 \
    --seq_len 64 \
    --skip 1 \
    --batch_size 128 \
    --num_iterations 10000 \
    --num_epochs 200 \
    --input_dim 1 \
    --noise_dim 10 \
    --ngf 64 \
    --ndf 64 \
    --learning_rate 2e-4 \
    --print_every 200 \
    --checkpoint_every 10000
```

## Evaluate

```bash
python evaluate.py \
    --model_path checkpoints \
    --num_samples 20
```

## Note

1. 下面兩種狀況會讓你的模型生得很好，但是這並不合理：
   - Mask的缺值不能用包含要預測的序列統計量去填補，這會導致你以為他生的很好
   - Data normalization要注意不能用包含要預測的序列去normalize，他會學到
