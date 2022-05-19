cd ../../

number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}


hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
    python -W ignore main.py \
    --drift=score_network \
    --dataset=circle \
    --prior=gaussian \
    --prior_sde=brownian \
    --batch_size=1500 \
    --num_epochs=20 \
    --learning_rate=1e-3 \
    --schedule_iter=5 \
    --num_steps=20 \
    --sigma=1e-5 \
    --max_gamma=1. \
    --initial_gamma=0.1 \
    --solver=$solver \
    --gpus="$number_gpus" \
    --scale=0.0 \
    --do_dsb=$do_dsb
    
