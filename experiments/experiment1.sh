cd ..
# hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --gpus '"device=0,1"' dlf28/pytorch_lightning\
hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="0,1" dlf28/pytorch_lightning\
	python main.py \
	  --drift=score_network \
	  --diffusion=scalar \
	  --dataset=toy_experiment_blobs_2d \
	  --prior=gaussian \
	  --prior_sde=brownian \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=30 \
	  --eval_frequency=100 \
	  --learning_rate=1e-2\
	  --num_steps=10 \
	  --delta_t=0.05 \
	  --num_iter=50 \
	  --sigma=0.001 \
	  --gpus=2
