### TRAINING ###
python train.py --config-name=VerbaLightGCN_config run_name=VerbaLightGCN

### EVALUATION ###
python evaluation.py --config-name=VerbaLightGCN_config run_name=VerbaLightGCN_Toys_dataset training.pretrain_path="ckpt/toys/VerbaLightGCN_k3/VerbaLightGCN_k3-epoch=04-val_f1=0.00.ckpt"

## RANKING EVALUATION ###
python eval_ranking.py --mode=aggregate_profile
python eval_ranking.py --mode=ranking