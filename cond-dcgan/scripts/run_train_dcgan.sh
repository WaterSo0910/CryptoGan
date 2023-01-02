python train.py \
    --datapath ../data/Binance_ETHUSDT_1h.csv \
    --model cond-dcgan \
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