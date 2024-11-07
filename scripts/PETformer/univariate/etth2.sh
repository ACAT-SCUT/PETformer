if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

model_name=PETformer

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seq_len=720
pred_len=96
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
  --d_model 512 \
  --ff_factor 2 \
  --n_heads 8 \
  --e_layers 4 \
  --dropout 0.5 \
  --win_len 48 \
  --win_stride 48 \
  --channel_attn 0 \
  --attn_type 0 \
  --des 'Exp' \
  --loss 'mae' \
  --patience 10 \
  --train_epochs 50\
  --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log

pred_len=192
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
  --d_model 512 \
  --ff_factor 2 \
  --n_heads 8 \
  --e_layers 4 \
  --dropout 0.5 \
  --win_len 48 \
  --win_stride 48 \
  --channel_attn 0 \
  --attn_type 0 \
  --des 'Exp' \
  --loss 'mae' \
  --patience 10 \
  --train_epochs 50\
  --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log


pred_len=336
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
  --d_model 512 \
  --ff_factor 2 \
  --n_heads 8 \
  --e_layers 4 \
  --dropout 0.5 \
  --win_len 48 \
  --win_stride 48 \
  --channel_attn 0 \
  --attn_type 0 \
  --des 'Exp' \
  --loss 'mae' \
  --patience 10 \
  --train_epochs 50\
  --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log

pred_len=720
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
  --n_heads 4 \
  --e_layers 4 \
  --dropout 0.5 \
  --win_len 48 \
  --win_stride 48 \
  --channel_attn 0 \
  --attn_type 0 \
  --des 'Exp' \
  --loss 'mae' \
  --patience 10 \
  --train_epochs 50\
  --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log
