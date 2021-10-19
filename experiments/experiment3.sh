cd ..
python main.py \
  --drift=nn_general \
  --diffusion=scalar \
  --dataset=double_well_right \
  --prior=double_well_left \
  --prior_sde=double_well \
  --prior_dist=gaussian \
  --batch_size=50 \
  --num_epochs=30 \
  --eval_frequency=100 \
  --learning_rate=1e-3\
  --num_steps=10 \
  --delta_t=0.05 \
  --num_iter=50 \
  --batch_repeats=1 \
  --sigma=0.00001 \
