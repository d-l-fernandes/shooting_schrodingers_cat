cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=score_network \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=50 \
	  --eval_frequency=20 \
	  --learning_rate=5e-5 \
	  --num_steps=10 \
	  --num_iter=10 \
	  --batch_repeats=50 \
	  --sigma=0.001 \
	  --max_gamma=0.2 \
	  --solver=rossler \
	  --hill_scale=5. \
    --gpus="$number_gpus"
