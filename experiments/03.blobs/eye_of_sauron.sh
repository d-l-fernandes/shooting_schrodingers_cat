cd ../../

number_gpus=$1
visible_devices=$2
solver=$3

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=blobs_2d \
    --prior=gaussian \
    --prior_sde=eye_of_sauron \
    --solver=$solver \
    --gpus="$number_gpus" \
    --prior_scale=0.05 \
    --normalize=False
    

   
