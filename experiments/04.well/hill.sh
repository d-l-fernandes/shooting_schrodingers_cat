cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
    --scale=$(bc  -l <<< '1/2') \
    --solver=$solver \
    --gpus="$number_gpus" \
    --use_sigma_prior=True \
    --do_dsb=$do_dsb \
    --normalize=False \
    --num_steps=25 \
    

   
