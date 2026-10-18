"""Evaluate CAP FIFO agents with several initialization configurations.

This is intentionally separate from ``test.py`` because importing ``test.py``
starts its full experiment immediately.  The simulator arguments from
``param.py`` are still available, so normal environment options can be passed
to this script as well.

Example:
    python simulator/test_agent_variants.py --agent all --variant_num_exp 3 \
        --pidle_range 0.0 1.0 0.05
"""

import argparse
import csv
import json
import os
import sys

# Parse options owned by this runner before importing param.py.  param.py
# parses sys.argv at import time, so leave its arguments untouched.
_variant_parser = argparse.ArgumentParser(add_help=False)
_variant_parser.add_argument(
    '--agent',
    choices=[
        'dynamic_b_cap_fifo',
        'lookahead_cap_fifo',
        'backfill_cap_fifo',
        'efficiency_cap_fifo',
        'weighted_cap_fifo',
        'all',
    ],
    default='all',
)
_variant_parser.add_argument(
    '--variant_result_file',
    default='./simulator/results/agent_variants.csv',
    help='CSV file for variant results',
)
_variant_parser.add_argument(
    '--variant_num_exp',
    type=int,
    default=None,
    help='Override --num_exp for this sweep',
)
_variant_args, _param_argv = _variant_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _param_argv

import numpy as np
import pandas as pd

from param import args
from spark_env.env import Environment
from agents.dynamic_b_cap_fifo_agent import DynamicBCapFIFOAgent
from agents.lookahead_cap_fifo_agent import LookaheadCapFIFOAgent
from agents.backfill_cap_fifo_agent import BackfillCapFIFOAgent
from agents.efficiency_cap_fifo_agent import EfficiencyCapFIFOAgent
from agents.weighted_cap_fifo_agent import CarbonPowerAgent as WeightedCapFIFOAgent


VARIANTS = {
    'dynamic_b_cap_fifo': [
        {'name': 'default', 'base_B': 20, 'max_B': 60,
         'max_consecutive_waits': 3},
        {'name': 'middle_waits', 'base_B': 20, 'max_B': 60,
         'max_consecutive_waits': 5},
        {'name': 'high_waits', 'base_B': 20, 'max_B': 60,
         'max_consecutive_waits': 8},
        {'name': 'low_max', 'base_B': 10, 'max_B': 30,
         'max_consecutive_waits': 3},
        {'name': 'high_max', 'base_B': 20, 'max_B': 90,
         'max_consecutive_waits': 3},
    ],
    'lookahead_cap_fifo': [
        {'name': 'default', 'horizon_slots': 6, 'max_slowdown': 1.25,
         'max_consecutive_waits': 3},
        {'name': 'short_horizon', 'horizon_slots': 3,
         'max_slowdown': 1.25, 'max_consecutive_waits': 3},
        {'name': 'long_horizon', 'horizon_slots': 12,
         'max_slowdown': 1.25, 'max_consecutive_waits': 3},
        {'name': 'strict_slowdown', 'horizon_slots': 6,
         'max_slowdown': 1.05, 'max_consecutive_waits': 3},
        {'name': 'lenient_slowdown', 'horizon_slots': 6,
         'max_slowdown': 1.5, 'max_consecutive_waits': 3},
    ],
    'backfill_cap_fifo': [
        {'name': 'default', 'max_backfill_work': 120000.0,
         'max_consecutive_waits': 3},
        {'name': 'aggressive_backfill', 'max_backfill_work': 300000.0,
         'max_consecutive_waits': 3},
        {'name': 'conservative_backfill', 'max_backfill_work': 30000.0,
         'max_consecutive_waits': 3},
        {'name': 'middle_waits', 'max_backfill_work': 120000.0,
         'max_consecutive_waits': 5},
        {'name': 'high_waits', 'max_backfill_work': 120000.0,
         'max_consecutive_waits': 8},
    ],
    'efficiency_cap_fifo': [
        {'name': 'default', 'arrival_tolerance': 60000,
         'max_consecutive_waits': 3},
        {'name': 'wide_arrival_window', 'arrival_tolerance': 180000,
         'max_consecutive_waits': 3},
        {'name': 'narrow_arival_window', 'arrival_tolerance': 20000,
         'max_consecutive_waits': 3},
        {'name': 'middle_waits', 'arrival_tolerance': 60000,
         'max_consecutive_waits': 5},
        {'name': 'high_waits', 'arrival_tolerance': 60000,
         'max_consecutive_waits': 8},
    ],
    'weighted_cap_fifo': [
        {'name': 'default', 'rho': 0.7, 'exec_lower_bound': 20,
         'max_consecutive_waits': 3},
        {'name': 'veryhigh_dynamic_weight', 'rho': 1.5,
         'exec_lower_bound': 20, 'max_consecutive_waits': 3},
        {'name': 'high_dynamic_weight', 'rho': 1.0,
         'exec_lower_bound': 20, 'max_consecutive_waits': 3},
        {'name': 'middle_dynamic_weight', 'rho': 0.5,
         'exec_lower_bound': 20, 'max_consecutive_waits': 3},
        {'name': 'low_dynamic_weight', 'rho': 0.2,
         'exec_lower_bound': 20, 'max_consecutive_waits': 3},
    ],
}

AGENT_CLASSES = {
    'dynamic_b_cap_fifo': DynamicBCapFIFOAgent,
    'lookahead_cap_fifo': LookaheadCapFIFOAgent,
    'backfill_cap_fifo': BackfillCapFIFOAgent,
    'efficiency_cap_fifo': EfficiencyCapFIFOAgent,
    'weighted_cap_fifo': WeightedCapFIFOAgent,
}


def load_carbon_schedule():
    frame = pd.read_csv(args.carbon_trace)
    start_rng = np.random.RandomState(args.seed)
    start = start_rng.randint(0, len(frame) - 100)
    carbon = frame['carbon_intensity_avg'].iloc[start:start + 100].tolist()
    return {60000 * index: value for index, value in enumerate(carbon)}


def create_power_models():
    """Create the same P_IDLE/P_DYN sweep used by ``test.py``."""
    pidle_start, pidle_stop, pidle_step = args.pidle_range
    if not 0 <= pidle_start <= 1 or not 0 <= pidle_stop <= 1:
        raise ValueError('--pidle_range START and STOP must be between 0 and 1')
    if pidle_step <= 0:
        raise ValueError('--pidle_range STEP must be greater than 0')
    if pidle_start > pidle_stop:
        raise ValueError('--pidle_range START must not be greater than STOP')

    pidle_values = np.arange(
        pidle_start, pidle_stop + pidle_step / 2, pidle_step)
    pidle_values = [
        round(float(value), 10)
        for value in pidle_values
        if 0 <= value <= 1
    ]
    return [
        {
            'name': f'model_{index}',
            'pidle': pidle,
            'pdyn': 1.0 - pidle,
        }
        for index, pidle in enumerate(pidle_values)
    ]


def run_agent(env, agent, scheme):
    observation = env.observe()
    total_reward = 0.0
    done = False
    stalled_steps = 0
    previous_time = env.wall_time.curr_time
    steps = 0

    while not done and steps < args.max_progress_points * args.progress_interval:
        node, use_exec, carbon_aware = agent.get_action(observation)
        if carbon_aware:
            observation, reward, done = env.step(
                node, use_exec, carbon_aware=True)
        else:
            observation, reward, done = env.step(
                node, use_exec, carbon_aware=False)
        total_reward += reward
        steps += 1

        current_time = env.wall_time.curr_time
        stalled_steps = stalled_steps + 1 if current_time == previous_time else 0
        previous_time = current_time
        if not done and stalled_steps >= args.max_stalled_steps:
            return total_reward, False, 'max_stalled_steps'

    if not done:
        return total_reward, False, 'max_progress_points'
    return total_reward, True, ''


def carbon_usage(env, carbon_schedule, pidle, pdyn):
    """Calculate dynamic and provisioned-idle carbon for a completed run."""
    default_carbon = carbon_schedule[next(iter(carbon_schedule))]
    dynamic = 0.0
    for job_dag in env.finished_job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                start = int(task.start_time)
                finish = int(task.finish_time)
                for key in range(start - start % 60000,
                                 finish - finish % 60000 + 60000, 60000):
                    end = min(finish, key + 60000)
                    duration = max(0, end - max(start, key))
                    dynamic += duration * pdyn * carbon_schedule.get(
                        key, default_carbon)

    idle = 0.0
    for key in range(0, int(env.wall_time.curr_time), 60000):
        duration = min(60000, int(env.wall_time.curr_time) - key)
        idle += args.exec_cap * pidle * duration * carbon_schedule.get(
            key, default_carbon)
    return dynamic, idle, dynamic + idle


def main():
    if args.seed is None:
        args.seed = 0
    if args.max_progress_points <= 0 or args.progress_interval <= 0:
        raise ValueError(
            '--max_progress_points and --progress_interval must be positive')

    carbon_schedule = load_carbon_schedule()
    power_models = create_power_models()
    agents = list(AGENT_CLASSES) if _variant_args.agent == 'all' \
        else [_variant_args.agent]
    num_exp = (_variant_args.variant_num_exp
               if _variant_args.variant_num_exp is not None else args.num_exp)
    if num_exp <= 0:
        raise ValueError('the number of experiments must be positive')

    seed_rng = np.random.RandomState(args.seed)
    experiment_seeds = seed_rng.randint(
        0, np.iinfo(np.int32).max, size=num_exp)
    results = []

    for scheme in agents:
        agent_class = AGENT_CLASSES[scheme]
        for variant in VARIANTS[scheme]:
            variant_values = {
                key: value for key, value in variant.items() if key != 'name'
            }
            for power_model in power_models:
                for experiment, experiment_seed in enumerate(
                        experiment_seeds, 1):
                    np.random.seed(int(experiment_seed))
                    env = Environment(carbon_schedule=carbon_schedule)
                    env.seed(int(experiment_seed))
                    env.reset()
                    constructor_values = {
                        **variant_values,
                        'pidle': power_model['pidle'],
                        'pdyn': power_model['pdyn'],
                    }
                    try:
                        agent = agent_class(
                            exec_cap=args.exec_cap,
                            carbon_schedule=carbon_schedule,
                            **constructor_values,
                        )
                        reward, completed, stop_reason = run_agent(
                            env, agent, scheme)
                    except ValueError as error:
                        reward = np.nan
                        completed = False
                        stop_reason = 'value_error: ' + str(error)

                    if completed:
                        dynamic, idle, total = carbon_usage(
                            env, carbon_schedule,
                            power_model['pidle'],
                            power_model['pdyn'],
                        )
                    else:
                        dynamic = idle = total = np.nan

                    results.append({
                        'agent': scheme,
                        'variant': variant['name'],
                        'power_model': power_model['name'],
                        'pidle': power_model['pidle'],
                        'pdyn': power_model['pdyn'],
                        'initialization': json.dumps(
                            constructor_values, sort_keys=True),
                        'experiment': experiment,
                        'completed': completed,
                        'stop_reason': stop_reason,
                        'wall_time': env.wall_time.curr_time,
                        'dynamic_carbon_usage': dynamic,
                        'idle_carbon_usage': idle,
                        'total_carbon_usage': total,
                    })
                    print(
                        f'{scheme}/{variant["name"]}/'
                        f'{power_model["name"]} '
                        f'experiment {experiment}: '
                        f'{"completed" if completed else stop_reason}',
                        flush=True,
                    )

    result_file = _variant_args.variant_result_file
    parent = os.path.dirname(result_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(result_file, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Wrote {len(results)} results to {result_file}')


if __name__ == '__main__':
    main()
