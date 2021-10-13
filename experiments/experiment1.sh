cd ..
python main.py \
  --drift=nn_general \
  --diffusion=constant_diagonal \
  --dataset=toy_experiment_blobs_2d \
  --prior=gaussian \
  --prior_sde=brownian \
  --prior_dist=learnable_gaussian \
  --batch_size=50 \
  --num_epochs=30 \
  --eval_frequency=20 \
  --learning_rate=1e-3\
  --num_steps=5 \
  --delta_t=0.2 \
  --num_iter=20 \
  --batch_repeats=10
