#!/usr/bin bash
for dataset in  'snips' # 'clinc' 'stackoverflow' 'snips'
do
  for known_cls_ratio in  1
  do
    for s in  0 1 2 3 4 5 6 7 8 9
    do
        python pretrain.py \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio 1.0 \
            --seed $s \
            --gpu_id 3
    done
  done
done