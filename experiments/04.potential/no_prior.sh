cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=nn_general \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
    --batch_size=1500 \
    --num_epochs=20 \
    --learning_rate=5e-4 \
    --schedule_iter=0 \
    --num_steps=25 \
    --sigma=1e-3 \
    --total_gamma=0.5 \
    --normalize=False \
    --solver=$solver \
    --gpus="$number_gpus" \
    --do_dsb=$do_dsb \
    --scale=0.0
    

   
