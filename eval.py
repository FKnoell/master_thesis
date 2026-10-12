import simexpal
import pathlib

config = simexpal.config_for_dir(".")

for experiment in config.all_experiments():
    for run in experiment.get_runs():
        output_dir = pathlib.Path(run.output_path)
        print(f"{run}: {output_dir}")