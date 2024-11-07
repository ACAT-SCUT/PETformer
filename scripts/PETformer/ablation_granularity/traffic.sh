if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PETformer

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom


seq_len=720
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_Point' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 12 \
      --ff_factor 2 \
      --n_heads 1 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 1 \
      --win_stride 1 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 3 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_Point'.log
done


for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_SubSeq6' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 64 \
      --ff_factor 2 \
      --n_heads 2 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 6 \
      --win_stride 6 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_SubSeq6'.log
done


for pred_len in 96 192 336
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_SubSeq12' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 128 \
      --ff_factor 2 \
      --n_heads 4 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 12 \
      --win_stride 12 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 6 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_SubSeq12'.log
done
for pred_len in 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_SubSeq12' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 128 \
      --ff_factor 2 \
      --n_heads 4 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 12 \
      --win_stride 12 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_SubSeq12'.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_SubSeq24' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 256 \
      --ff_factor 2 \
      --n_heads 8 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 24 \
      --win_stride 24 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 6 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_SubSeq24'.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_SubSeq48' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --d_model 512 \
      --ff_factor 2 \
      --n_heads 16 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 48 \
      --win_stride 48 \
      --channel_attn 0 \
      --attn_type 0 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 5\
      --train_epochs 25\
      --itr 1 --batch_size 6 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_SubSeq48'.log
done
