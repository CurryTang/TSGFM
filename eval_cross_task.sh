# CUDA_VISIBLE_DEVICES=0 bash llm_eval.sh cora nc ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed.3-nc-lp-projector/ citationcross
# CUDA_VISIBLE_DEVICES=0 bash llm_eval.sh citeseer nc ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed.3-nc-lp-projector/ citationcross
# CUDA_VISIBLE_DEVICES=0 bash llm_eval.sh pubmed nc ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed.3-nc-lp-projector/ citationcross
# bash llmres.sh cora nc ./checkpoints/cora_nc_citationcross
# bash llmres.sh citeseer nc ./checkpoints/citeseer_nc_citationcross
# bash llmres.sh pubmed nc ./checkpoints/pubmed_nc_citationcross

CUDA_VISIBLE_DEVICES=0 bash llm_eval_link.sh cora lp ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed-nc-lp-projector/ citationcross
CUDA_VISIBLE_DEVICES=0 bash llm_eval_link.sh citeseer lp ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed-nc-lp-projector/ citationcross
CUDA_VISIBLE_DEVICES=0 bash llm_eval_link.sh pubmed lp ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed-nc-lp-projector/ citationcross
bash llmres.sh cora lp ./checkpoints/cora_lp_citationcross
bash llmres.sh citeseer lp ./checkpoints/citeseer_lp_citationcross
bash llmres.sh pubmed lp ./checkpoints/pubmed_lp_citationcross