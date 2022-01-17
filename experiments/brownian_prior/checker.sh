cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
  python -W ignore main.py \
    --drift=score_network \
    --diffusion=scalar \
    --dataset=checker \
    --prior=gaussian \
    --prior_sde=brownian \
    --prior_dist=gaussian \
    --batch_size=1500 \
    --num_epochs=40 \
    --eval_frequency=20 \
    --learning_rate=5e-5 \
    --num_steps=20 \
    --num_iter=10 \
	  --batch_repeats=40 \
    --sigma=0.001 \
    --max_gamma=2. \
	  --solver=rossler \
    --gpus="$number_gpus"
