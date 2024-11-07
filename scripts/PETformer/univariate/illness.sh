if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

model_name=Timeformer

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom


seq_len=72
for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --d_model 256 \
      --ff_factor 2 \
      --n_heads 8 \
      --e_layers 4 \
      --dropout 0.25 \
      --win_len 12 \
      --win_stride 6 \
      --channel_attn 0 \
      --attn_type 1 \
      --des 'Exp' \
      --loss 'mae' \
      --patience 10 \
      --train_epochs 50\
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log
done
