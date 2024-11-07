if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=PETformer

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=720
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_EncDecTransformer' \
      --model EncDecTransformer \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --d_model 512 \
      --ff_factor 2 \
      --n_heads 8 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 48 \
      --win_stride 48 \
      --channel_attn 1 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 10\
      --train_epochs 50\
      --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_EncDecTransformer'.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_FeatureHeadTransformer' \
      --model FeatureHeadTransformer \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --d_model 512 \
      --ff_factor 2 \
      --n_heads 8 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 48 \
      --win_stride 48 \
      --channel_attn 1 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 10\
      --train_epochs 50\
      --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_FeatureHeadTransformer'.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_FlattenTransformer' \
      --model FlattenTransformer \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --d_model 512 \
      --ff_factor 2 \
      --n_heads 8 \
      --e_layers 4 \
      --dropout 0.5 \
      --win_len 48 \
      --win_stride 48 \
      --channel_attn 1 \
      --des 'Exp' \
      --loss 'smooth' \
      --patience 10\
      --train_epochs 50\
      --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_FlattenTransformer'.log
done

for attn_type in 0 1 2 3
do
    for pred_len in 96 192 336 720
    do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name'_'$seq_len'_'$pred_len'_attnType'$attn_type \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --d_model 512 \
          --ff_factor 2 \
          --n_heads 8 \
          --e_layers 4 \
          --dropout 0.5 \
          --win_len 48 \
          --win_stride 48 \
          --channel_attn 1 \
          --attn_type $attn_type \
          --des 'Exp' \
          --loss 'smooth' \
          --patience 10\
          --train_epochs 50\
          --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_attnType'$attn_type.log
    done
done


