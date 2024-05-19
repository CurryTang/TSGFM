for d in "bookhis" "bookchild" "elecomp" "elephoto" "sportsfit" "products"
do 
    CUDA_VISIBLE_DEVICES=7 python3 run_cdm.py task_names $d d_min_ratio 1 d_multiple 1 exp_name "single_${d}" model ofamlp
    CUDA_VISIBLE_DEVICES=7 python3 run_cdm.py task_names $d d_min_ratio 1 d_multiple 1 exp_name "single_${d}" model noparam
done 