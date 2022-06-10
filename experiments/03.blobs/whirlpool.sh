cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=blobs_2d \
    --prior=gaussian \
    --prior_sde=whirlpool \
    --solver=$solver \
    --gpus="$number_gpus" \
    --prior_scale=0.05 \
    --normalize=False \
    --do_dsb=$do_dsb

   
