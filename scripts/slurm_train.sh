#!/bin/bash
#SBATCH
#SBATCH -n 4     # 指定核心数量
#SBATCH -N 1     # 指定node的数量
#SBATCH --ntasks-per-node=4
#SBATCH -p project # 提交到哪一个分区
#SBATCH --mem=64G # 所有核心可以使用的内存池大小
#SBATCH --gres=gpu:4 # 需要使用多少GPU，n是需要的数量
#SBATCH -t 3-00:00:00 #Maximum runtime of 48 hours
#SBATCH --job-name=c_1_2


module purge
module load Anaconda3
source activate whn
module load cuda11.8
module load slurm


now=$(date +"%Y%m%d_%H%M%S")



# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['baseline_sup', 'baseline_semi', 'allspark']
# split:
## 1. for 'pascal' select:
###  - original ['92', '183', '366', '732' ,'1464']
###  - augmented ['1_16', '1_8', '1_4', '1_2']
###  - augmented (U2PL splits) ['u2pl_1_16', 'u2pl_1_8', 'u2pl_1_4', 'u2pl_1_2']
## 2. for 'cityscapes' select: ['1_16', '1_8', '1_4', '1_2']
## 3. for 'coco' select: ['1_512', '1_256', '1_128', '1_32']


dataset='cityscapes'
method='allspark'
split='1_2'


config=configs/${dataset}_${method}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$split


mkdir -p $save_path



srun --mpi=pmi2 \
    python -u \
    train_$method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
