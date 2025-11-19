python -m app.train.training_hpo \
    --n_trials 10 \
    --study_name "dogs-efficientnet-hpo-v1" \
    --optuna_db "optuna_dogs.db"

# --n_trials null \
# --redo_trial 14 \