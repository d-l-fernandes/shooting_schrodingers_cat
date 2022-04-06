cd ../../

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=nn_general_mnist \
	  --diffusion=scalar \
	  --dataset=mnist \
	  --prior=gaussian \
	  --variational=gaussian_mnist \
	  --prior_sde=brownian \
	  --prior_dist=gaussian \
	  --batch_size=500 \
	  --num_epochs=50 \
	  --eval_frequency=10 \
	  --learning_rate=5e-5 \
	  --num_steps=10 \
	  --num_iter=5 \
	  --batch_repeats=50 \
	  --num_samples=5 \
	  --sigma=0.001 \
	  --solver=rossler \
    --gpus="$number_gpus"
