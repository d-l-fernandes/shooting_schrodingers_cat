cd ..
hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --gpus '"device=0,1"' dlf28/pytorch_lightning\
	python main.py \
	  --drift=nn_general \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=30 \
	  --eval_frequency=100 \
	  --learning_rate=1e-3\
	  --num_steps=10 \
	  --delta_t=0.05 \
	  --num_iter=50 \
	  --sigma=0.001 \
	  --max_gamma=0.5 \
	  --gpus=2