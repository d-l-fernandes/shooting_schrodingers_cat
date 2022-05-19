cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}
scale=$5


hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=nn_general \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
    --batch_size=1500 \
    --num_epochs=20 \
    --learning_rate=1e-3 \
    --schedule_iter=0 \
    --num_steps=25 \
    --sigma=1e-3 \
    --max_gamma=1.0 \
    --initial_gamma=0.1 \
    --normalize=False \
    --solver=$solver \
    --gpus="$number_gpus" \
    --do_dsb=$do_dsb \
    --scale=$scale
    

   
