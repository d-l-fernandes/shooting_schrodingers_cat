cd ..

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
  python main.py \
    --drift=score_network \
    --diffusion=scalar \
    --dataset=circle \
    --prior=gaussian \
    --prior_sde=brownian \
    --prior_dist=gaussian \
    --batch_size=1500 \
    --num_epochs=25 \
    --eval_frequency=200 \
    --learning_rate=1e-3 \
    --num_steps=10 \
    --delta_t=0.05 \
    --num_iter=100 \
    --sigma=0.001 \
    --gpus="$number_gpus"
