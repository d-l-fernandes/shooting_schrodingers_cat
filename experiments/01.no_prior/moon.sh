cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=moon \
    --prior=gaussian \
    --prior_sde=brownian \
    --solver=$solver \
    --gpus="$number_gpus" \
    --scale=0.0 \
    --do_dsb=$do_dsb
    
