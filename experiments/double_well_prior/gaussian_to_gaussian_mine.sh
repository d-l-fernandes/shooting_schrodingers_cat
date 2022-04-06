cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=nn_general \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=25 \
	  --eval_frequency=50 \
	  --learning_rate=5e-4 \
	  --schedule_iter=0 \
	  --num_steps=30 \
	  --num_iter=25 \
	  --batch_repeats=20 \
	  --initial_sigma=1e-5 \
	  --min_sigma=1e-5 \
	  --total_gamma=1. \
	  --normalize=False \
	  --solver=srk \
    --gpus="$number_gpus"
