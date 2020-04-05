
# $1 lr 5e-4     0.0005, 0.1 0.001 0.0001 0.00001
# dim=16

task_i=0
for dim in $(<./params/params_dim.txt)
do
  for lr_weight_decay in $(<./params/params_lr_weight_decay.txt)
  do
    task_i=`expr $task_i + 1`
#    if [ $task_i -lt 24 ]
#    then
#      continue
#    fi
    lr=`echo $lr_weight_decay | cut -d \, -f 1`
    l2_weight_decay=`echo $lr_weight_decay | cut -d \, -f 2`
    echo '>>>Task:'$task_i 'dim='$dim 'lr='$lr 'l2_weight_decay='$l2_weight_decay
    python3 main.py --dataset music -m hgcn --dim $dim --hidden1 64 --hidden2 32  --batch_size 256 --l2_weight_decay $l2_weight_decay --lr $lr --mode train
  done
done