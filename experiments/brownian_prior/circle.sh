cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=score_network \
	  --dataset=circle \
	  --prior=gaussian \
	  --prior_sde=brownian \
	  --prior_dist=learnable_gaussian \
	  --batch_size=1500 \
	  --num_epochs=20 \
	  --eval_frequency=50 \
	  --learning_rate=1e-4 \
	  --num_steps=10 \
	  --num_iter=25 \
	  --batch_repeats=20 \
	  --initial_sigma=0.001 \
	  --total_gamma=1. \
	  --solver=rossler \
    --gpus="$number_gpus"
