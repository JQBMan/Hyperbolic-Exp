# Example: (./run.sh &) Warning: used under src
# music
echo 'Current Direction:'
pwd
echo 'Starting Work...'
#python3 main.py --dataset music -m embedding --dim 16 --hidden1 64 --hidden2 32 --batch_size 512 --l2_weight_decay 1e-4 --lr 5e-4 --mode $1
#python3 main.py --dataset music -m mlp --dim 16 --hidden1 64 --hidden2 32 --batch_size 512 --l2_weight_decay 1e-4 --lr 5e-4 --mode $1
#python3 main.py --dataset music -m gcn --dim 16 --hidden1 64 --hidden2 32 --batch_size 512 --l2_weight_decay 1e-4 --lr 5e-4 --mode $1
#python3 main.py --dataset music -m gat --dim 16 --hidden1 64 --hidden2 32  --batch_size 512 --l2_weight_decay 1e-4 --lr 5e-4 --mode $1
#python3 main.py --dataset music -m hnn --dim 16 --hidden1 64 --hidden2 32  --batch_size 512 --l2_weight_decay 1e-4 --lr 5e-4 --mode $1
python3 main.py --dataset music -m $1 --dim 16 --hidden1 64 --hidden2 32  --batch_size 64 --l2_weight_decay 2e-4 --lr 1e-4 --mode $2
#python3 main.py --dataset music -m hgat --dim 16 --hidden1 64 --hidden2 32  --batch_size 512 --l2_weight_decay 2e-4 --lr 5e-4 --mode $1
echo 'Finished.'


