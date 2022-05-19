cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}


hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=nn_general \
    --dataset=blobs_2d \
    --prior=gaussian \
    --prior_sde=whirlpool \
    --batch_size=1500 \
    --num_epochs=20 \
    --learning_rate=1e-3 \
    --schedule_iter=5 \
    --num_steps=20 \
    --sigma=1e-3 \
    --max_gamma=0.5 \
    --initial_gamma=0.1 \
    --solver=$solver \
    --gpus="$number_gpus" \
    --scale=1.0 \
    --prior_scale=0.05 \
    --normalize=False \
    --do_dsb=$do_dsb

   
