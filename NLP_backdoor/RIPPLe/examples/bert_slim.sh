# export MKL_SERVICE_FORCE_INTEL=1
export PYTHONPATH=../..:$PYTHONPATH


CUDA_VISIBLE_DEVICES=$1 \
python slim_batch_experiments.py \
batch \
--manifesto bert_mani/ncprune_badnet.yaml

