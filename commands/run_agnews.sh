task=agnews 
gpu=0
n_gpu=2

train_seed=42
label_per_class=30
model_type=roberta-base
train_seed=${train_seed}
method=train
max_seq_len=128
self_training_batch_size=32
eval_batch_size=256
dev_labels=100
steps=100
logging_steps=10
st_logging_steps=20
epochs=18
k=5

lr=2e-5
eps=0.9
self_training_weight=1
gce_loss_q=0.6
lr_st=1e-5
batch_size=8
self_training_batch_size=32
self_training_update_period=1000
self_training_max_step=2000
num_unlabeled=2000
num_unlabeled_add=2000
ssl_cmd="--learning_rate_st=${lr_st} --self_training_eps=${eps} --self_training_weight=${self_training_weight} --self_training_update_period=${self_training_update_period} --gce_loss_q=${gce_loss_q} --num_unlabeled=${num_unlabeled} --num_unlabeled_add=${num_unlabeled_add}"


model_type=${model_type} #dmis-lab/biobert-v1.1 #"allenai/scibert_scivocab_uncased"
output_dir=${task}/${label_per_class}/model #../datasets/${task}-${label_per_class}-10/model
mkdir -p ${output_dir}
echo ${method}
mkdir -p ../datasets/${task}-${label_per_class}/cache
# valid_${train_label}.json
train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --do_train --do_eval --task=${task} \
	--train_file=train.json --dev_file=valid.json --test_file=test.json \
	--unlabel_file=unlabeled.json \
	--data_dir=../datasets/${task}-${label_per_class} --train_seed=${train_seed} \
	--cache_dir="../datasets/${task}-${label_per_class}/cache" \
	--output_dir=${output_dir} \
	--logging_steps=${logging_steps} --self_train_logging_steps=${st_logging_steps} --dev_labels=${dev_labels} \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} --weight_decay=1e-8 \
	--learning_rate=${lr}  \
	--method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--self_training_batch_size=${self_training_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type} \
	--self_training_max_step=${self_training_max_step} \
	--sample_labels=${train_label} ${ssl_cmd} --k=${k} --label_per_class=${label_per_class}"
echo $train_cmd
eval $train_cmd
