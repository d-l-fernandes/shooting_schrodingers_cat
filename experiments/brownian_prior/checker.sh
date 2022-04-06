cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=score_network \
	  --dataset=checker \
	  --prior=gaussian \
	  --prior_sde=brownian \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=20 \
	  --eval_frequency=50 \
	  --learning_rate=1e-3 \
	  --schedule_iter=5 \
	  --num_steps=20 \
	  --num_iter=25 \
	  --batch_repeats=20 \
	  --initial_sigma=1e-5 \
	  --min_sigma=1e-5 \
	  --max_gamma=1. \
	  --initial_gamma=0.1 \
	  --solver=srk \
    --gpus="$number_gpus"
