# python3 pretrain_graphcl.py --dataset pcba,tox21,bace,bbbp,muv,toxcast,hiv --epoch 61 --save_epoch 10
# python3 finetune_graphcl.py --dataset tox21 --input_model_file /egr/research-dselab/chenzh85/nips/MyOFA/models_graphcl/graphcl_tox21_element20.pth
# python3 pretrain_graphcl.py --dataset tox21,bace,bbbp,muv,toxcast,hiv --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset tox21 --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset bace --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset bbbp --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset muv --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset toxcast --epoch 61 --save_epoch 10
# python3 pretrain_graphcl.py --dataset hiv --epoch 61 --save_epoch 10

for data in pcba
do
    python3 finetune_graphcl.py --dataset $data --input_model_file /egr/research-dselab/chenzh85/nips/MyOFA/models_graphcl/graphcl_pcba,tox21,bace,bbbp,muv,toxcast,hiv_element60.pth
    # python3 finetune_graphcl.py --dataset $data --input_model_file /egr/research-dselab/chenzh85/nips/MyOFA/models_graphcl/graphcl_tox21,bace,bbbp,muv,toxcast,hiv_element60.pth
done