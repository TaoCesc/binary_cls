#!/usr/bin bash
for dataset in  'banking' 'clinc' 'stackoverflow' 'snips'
do
  for known_cls_ratio in  1
  do
    for s in  8
    do
      for t in 0.0 1.0 2.0 3.0
      do
        for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
        do
          for m in 0.1 0.2 0.3 0.4
          do
            for r in 10. 20. 30. 40. 50.
            do
              python pretrain.py \
                  --dataset $dataset \
                  --known_cls_ratio $known_cls_ratio \
                  --labeled_ratio 1.0 \
                  --seed $s \
                  --gpu_id 3 \
                  --alpha $alpha \
                  --m $m \
                  --r $r \
                  --t $t
            done
          done
        done
      done
    done
  done
done