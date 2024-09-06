DATASET_NAME=$1
# PRED=$2
DEBUG=$2
CKPT=$3


source ./args.sh $DATASET_NAME $PRED

FREQ_SAVE_ITER=20000
CUDA_VISIBLE_DEVICES=1 python train/train_mrm.py --exp=$EXP \
  --batch_size $BS --lr 0.0001 --weight_decay 0.0  --num_workers=$NGPU \
      ${CKPT:+ --resume_checkpoint="${CKPT}"}  ${DEBUG:+ --debug="${DEBUG}"} \
      --recycle_data_path $RECYCLE_DATA_PATH --retarget_data_path $RETARGET_DATA_PATH ${HUMAN_DATA_PATH:+ --human_data_path="${HUMAN_DATA_PATH}"} \
      --normalize=$NORMALIZE --overlap=$OVERLAP --overwrite --log_interval=250
