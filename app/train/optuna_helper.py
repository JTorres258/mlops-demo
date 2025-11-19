from absl import app, flags
import optuna
from optuna.trial import TrialState
from pathlib import Path
import tensorflow as tf
from collections import defaultdict

from app.train.config import load_config

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "check_states", False, "Check trials with state: RUNNING, FAIL, COMPLETE."
)
flags.DEFINE_integer("trial_id", None, "Trial number to evaluate.")
flags.DEFINE_string("change_to_state", None, "Change trial state.")

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Global dataset variables (to be initialized in main)
base_cfg = load_config("./configs/base_config.yml")  # or your current config


def main(argv):

    del argv  # Unused

    optuna_db_path = Path(FLAGS.optuna_db).resolve()

    study = optuna.load_study(
        study_name="dogs-efficientnet-hpo-v1",
        storage=f"sqlite:///{optuna_db_path}",
    )

    # Check trials with given state
    if FLAGS.check_states:

        groups = defaultdict(list)

        for t in study.trials:
            groups[t.state].append(t.number)

        print("\n=== Trial States Summary ===")
        for state in optuna.trial.TrialState:
            trial_nums = groups.get(state, [])
            print(f"{state.name:10} → {trial_nums}")
        print("============================\n")

    if FLAGS.change_to_state:

        STATE_MAP = {
            "FAIL": TrialState.FAIL,
            "RUNNING": TrialState.RUNNING,
            "COMPLETE": TrialState.COMPLETE,
            "PRUNED": TrialState.PRUNED,
            "WAITING": TrialState.WAITING,
        }

        state_enum = STATE_MAP[FLAGS.change_to_state.upper()]

        assert FLAGS.trial_id is not None, "Must provide --trial_id to change state."

        trial = study.trials[FLAGS.trial_id]
        old_state = trial.state

        study._storage._backend.set_trial_state_values(
            trial_id=trial._trial_id,
            state=state_enum,
            values=None,  # or [] – we don't have a valid objective
        )

        print(
            f"Updated state in trial {FLAGS.trial_id}: from {old_state} to {study.trials[FLAGS.trial_id].state}"
        )


if __name__ == "__main__":
    app.run(main)
