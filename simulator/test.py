import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from spark_env.env import Environment
from agents.spark_agent import SparkAgent
from agents.heuristic_agent import DynamicPartitionAgent
from agents.carbon_aware_heuristic_agent import CarbonPartitionAgent
from agents.actor_agent import ActorAgent
from agents.carbon_aware_actor_agent import CarbonActorAgent
from agents.pcaps_actor_agent import PCAPSAgent
from agents.carbon_aware_fifo_agent import CarbonAgent
from agents.green_hadoop_agent import GreenHadoopThetaAgent
from spark_env.canvas import *
from param import *
from utils import *
import pandas as pd

# Always-on power models.
# P_IDLE is charged for every provisioned executor for the full schedule.
# P_DYN is charged for task execution time.
pidle_start, pidle_stop, pidle_step = args.pidle_range
if not 0 <= pidle_start <= 1 or not 0 <= pidle_stop <= 1:
    raise ValueError('--pidle_range START and STOP must be between 0 and 1')
if pidle_step <= 0:
    raise ValueError('--pidle_range STEP must be greater than 0')
if pidle_start > pidle_stop:
    raise ValueError('--pidle_range START must not be greater than STOP')

pidle_values = np.arange(
    pidle_start, pidle_stop + (pidle_step / 2), pidle_step)
pidle_values = [round(float(value), 10) for value in pidle_values
                if 0 <= value <= 1]

power_models = [
    {"name": f"model_{i}", "pidle": pidle, "pdyn": 1.0 - pidle}
    for i, pidle in enumerate(pidle_values)
]

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
tf.compat.v1.set_random_seed(args.seed)

df = pd.read_csv(args.carbon_trace)
c = df["carbon_intensity_avg"]
r = df['power_production_percent_renewable_avg']

# pick a random start time in the trace
start_time = np.random.randint(0, len(c) - 100)
c = c[start_time:start_time + 100].to_list()
r = r[start_time:start_time + 100].to_list()

# # Select a specific start index for the trace to ensure reproducibility
# TRACE_START = 500
# 
# if TRACE_START < 0 or TRACE_START + 100 > len(c):
#     raise ValueError("TRACE_START must select 100 valid samples")
# 
# c = c[TRACE_START:TRACE_START + 100].to_list()
# r = r[TRACE_START:TRACE_START + 100].to_list()

carbon_schedule = [(60000 * i, c[i]) for i in range(len(c))]

carbon_dict = {}
for i in range(len(c)):
    carbon_dict[60000*i] = c[i]

renewable_dict = {}
for i in range(len(r)):
    renewable_dict[60000*i] = r[i]

def create_agent(scheme, power_model):
    """Create a fresh scheduler configured for one power model."""
    pidle = power_model['pidle']
    pdyn = power_model['pdyn']

    if scheme in ('pcaps', 'cap_decima'):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(args.seed)
        sess = tf.compat.v1.Session()
        agent_class = PCAPSAgent if scheme == 'pcaps' else CarbonActorAgent
        return agent_class(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1), carbon_dict,
            pidle=pidle, pdyn=pdyn)

    if scheme == 'decima':
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(args.seed)
        sess = tf.compat.v1.Session()
        return ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1), pidle=pidle, pdyn=pdyn)
    if scheme == 'dynamic_partition':
        return DynamicPartitionAgent()
    if scheme == 'spark_fifo':
        return SparkAgent(exec_cap=args.exec_cap)
    if scheme == 'cap_fifo':
        return CarbonAgent(
            exec_cap=args.exec_cap, carbon_schedule=carbon_dict,
            pidle=pidle, pdyn=pdyn)
    if scheme == 'cap_partition':
        return CarbonPartitionAgent(
            exec_cap=args.exec_cap, carbon_schedule=carbon_dict,
            pidle=pidle, pdyn=pdyn)
    if scheme == 'green_hadoop':
        return GreenHadoopThetaAgent(
            exec_cap=args.exec_cap, renewable_dict=renewable_dict)

    raise ValueError('scheme ' + str(scheme) + ' not recognized')


def run_scheme(env, scheme, agent):
    """Run one scheduler against one fresh environment."""
    obs = env.observe()
    total_reward = 0
    done = False
    step = 0

    while not done:
        if step % 10 == 0:
            print('.', end='', flush=True)
        step += 1

        if scheme in ('pcaps', 'cap_decima', 'cap_fifo',
                      'cap_partition', 'green_hadoop'):
            node, use_exec, carbon_aware = agent.get_action(obs)
            obs, reward, done = env.step(
                node, use_exec, carbon_aware=carbon_aware)
        else:
            node, use_exec = agent.get_action(obs)
            obs, reward, done = env.step(node, use_exec)

        total_reward += reward

    return total_reward


def calculate_carbon_usage(env, power_model):
    """Calculate carbon usage for the schedule and one power model."""
    default_carbon_value = carbon_dict[next(iter(carbon_dict))]
    schedule_end = int(env.wall_time.curr_time)
    pidle = power_model['pidle']
    pdyn = power_model['pdyn']

    total_dynamic_carbon_usage = 0.0
    total_idle_carbon_usage = 0.0

    for job_dag in env.finished_job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                start = int(task.start_time)
                finish = int(task.finish_time)
                start_key = start - (start % 60000)
                end_key = finish - (finish % 60000)

                for key in range(start_key, end_key + 1, 60000):
                    carbon_value = carbon_dict.get(key, default_carbon_value)
                    bucket_start = max(start, key)
                    bucket_end = min(finish, key + 60000)
                    duration = bucket_end - bucket_start
                    total_dynamic_carbon_usage += (
                        duration * pdyn * carbon_value)

    for key in range(0, schedule_end, 60000):
        bucket_end = min(key + 60000, schedule_end)
        duration = bucket_end - key
        carbon_value = carbon_dict.get(key, default_carbon_value)
        total_idle_carbon_usage += (
            args.exec_cap * pidle * duration * carbon_value)

    return (
        total_dynamic_carbon_usage + total_idle_carbon_usage,
        total_dynamic_carbon_usage,
        total_idle_carbon_usage,
    )

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


carbon_power_results = []

scheme_results_by_model = {
    model["name"]: []
    for model in power_models
}

for model in power_models:
    print('Power model ' + model['name'])

    for exp in range(args.num_exp):
        print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

        for scheme in args.test_schemes:
            print('Scheme ' + scheme)
            env = Environment(carbon_schedule=carbon_dict)
            env.seed(args.num_ep + exp)
            env.reset()
            agent = create_agent(scheme, model)
            total_reward = run_scheme(env, scheme, agent)
            all_total_reward[scheme].append(total_reward)

            job_durations = [
                job_dag.completion_time - job_dag.start_time
                for job_dag in env.finished_job_dags
            ]
            total_carbon_usage, total_dynamic_carbon_usage, \
                total_idle_carbon_usage = calculate_carbon_usage(env, model)

            scheme_results_by_model[model["name"]].append((
                scheme,
                env.wall_time.curr_time,
                total_carbon_usage,
                np.mean(job_durations)
            ))

            carbon_power_results.append({
                "experiment": exp + 1,
                "scheme": scheme,
                "power_model": model["name"],
                "pidle": model['pidle'],
                "pdyn": model['pdyn'],
                "dynamic_carbon_usage": total_dynamic_carbon_usage,
                "idle_carbon_usage": total_idle_carbon_usage,
                "total_carbon_usage": total_carbon_usage,
            })

            print("")
            print(
                f"Carbon usage — experiment {exp + 1}, "
                f"scheme {scheme}, model {model['name']}: "
                f"dynamic={total_dynamic_carbon_usage:.2f}, "
                f"idle={total_idle_carbon_usage:.2f}, "
                f"total={total_carbon_usage:.2f}"
            )
        
    '''
    print("Creating graphs\n")
    if args.canvs_visualization == 0: 
        visualize_carbon_usage_aggregated(
            scheme_results,
            args.result_folder + 'aggregated_carbon_usage.png'
        )

    elif args.canvs_visualization == 1:
        visualize_dag_time_save_pdf(
            env.finished_job_dags, env.executors,
            args.result_folder + 'visualization_exp_' + \
            str(exp) + '_scheme_' + scheme + \
            '.png', plot_type='app')
    elif args.canvs_visualization == 2:
        visualize_executor_usage(env.finished_job_dags,
            args.result_folder + 'visualization_exp_' + \
            str(exp) + '_scheme_' + scheme + '.png', carbon_dict)
    '''

    # # plot CDF of performance
    # if args.canvs_visualization == 0:
    #     visualize_carbon(carbon_time_list)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # for scheme in args.test_schemes:
    #     x, y = compute_CDF(all_total_reward[scheme])
    #     ax.plot(x, y)

    # plt.xlabel('Total reward')
    # plt.ylabel('CDF')
    # plt.legend(args.test_schemes)
    # fig.savefig(args.result_folder + 'total_reward.png')

    # plt.close(fig)

pd.DataFrame(carbon_power_results).to_csv(
    args.result_folder + "carbon_power_models.csv",
    index=False
)