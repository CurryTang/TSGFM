CUDA_VISIBLE_DEVICES=2 python3 run_cdm.py --override e2e_all_config.yaml model ofa exp_name all_ofa num_epochs 30 & 
CUDA_VISIBLE_DEVICES=3 python3 run_cdm.py --override e2e_all_config.yaml model ofamlp exp_name all_ofamlp num_epochs 30 & 
CUDA_VISIBLE_DEVICES=4 python3 run_cdm.py --override e2e_all_config.yaml model adapool exp_name all_adapool num_epochs 30 & 
CUDA_VISIBLE_DEVICES=5 python3 run_cdm.py --override e2e_all_config.yaml model noparam exp_name all_noparam num_epochs 30 & 
CUDA_VISIBLE_DEVICES=6 python3 run_cdm.py --override e2e_all_config.yaml model ofa exp_name all_ofa llm_name e5 & 
CUDA_VISIBLE_DEVICES=7 python3 run_cdm.py --override e2e_all_config.yaml model ofa exp_name all_ofa llm_name e5_mistral &
