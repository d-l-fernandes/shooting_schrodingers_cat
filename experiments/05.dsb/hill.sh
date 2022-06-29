cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
    --do_dsb=True \
    --gpus="$number_gpus" \
    --normalize=False \
    --num_steps=100 \
    

   
