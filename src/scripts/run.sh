# Example: (./run.sh &) !!!Warning: used under src
#
# movie1m --hidden1 128 --hidden2 64
# book  --hidden1 64 --hidden2 16
# musice  --hidden1 64 --hidden2 32
pwd
(sudo python3 main.py --dataset $1 --model $2 --dim $3 --hidden1 $4 --hidden2 $5 --batch_size $6 --l2_weight_decay $7 --lr $8 &)