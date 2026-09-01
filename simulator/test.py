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
power_models = [
    {"name": "model_1", "pidle": 0.0, "pdyn": 1.0},
    {"name": "model_2", "pidle": 0.2, "pdyn": 0.8},
    {"name": "model_3", "pidle": 0.5, "pdyn": 0.5},
    {"name": "model_4", "pidle": 0.8, "pdyn": 0.2},
    {"name": "model_5", "pidle": 1.0, "pdyn": 0.0}
]

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
tf.compat.v1.set_random_seed(args.seed)

df = pd.read_csv(args.carbon_trace)
c = df["carbon_intensity_avg"]
r = df['power_production_percent_renewable_avg']

# # pick a random start time in the trace
# start_time = np.random.randint(0, len(c) - 100)
# c = c[start_time:start_time + 100].to_list()
# r = r[start_time:start_time + 100].to_list()

# Select a specific start index for the trace to ensure reproducibility
TRACE_START = 500

if TRACE_START < 0 or TRACE_START + 100 > len(c):
    raise ValueError("TRACE_START must select 100 valid samples")

c = c[TRACE_START:TRACE_START + 100].to_list()
r = r[TRACE_START:TRACE_START + 100].to_list()

carbon_schedule = [(60000 * i, c[i]) for i in range(len(c))]

carbon_dict = {}
for i in range(len(c)):
    carbon_dict[60000*i] = c[i]

renewable_dict = {}
for i in range(len(r)):
    renewable_dict[60000*i] = r[i]

# set up environment
env = Environment(carbon_schedule=carbon_dict)

# set up agents
agents = {}
carbon_time_list = []

for scheme in args.test_schemes:
    if scheme == 'decima':
        sess = tf.compat.v1.Session()
        agents[scheme] = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
    elif scheme == 'pcaps' or scheme == 'cap_decima':
        agents[scheme] = None
    elif scheme == 'dynamic_partition':
        agents[scheme] = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agents[scheme] = SparkAgent(exec_cap=args.exec_cap)
    elif scheme == 'cap_fifo':
        agents[scheme] = CarbonAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    elif scheme == 'cap_partition':
        agents[scheme] = CarbonPartitionAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    elif scheme == 'green_hadoop':
        agents[scheme] = GreenHadoopThetaAgent(exec_cap=args.exec_cap, renewable_dict=renewable_dict)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


carbon_power_results = []

scheme_results_by_model = {
    model["name"]: []
    for model in power_models
}

for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        print('Scheme ' + scheme)
        # # reset environment with seed
        # env.seed(args.num_ep + exp)
        # env.reset()
        
        # Use a fixed seed for reproducibility across experiments
        FIXED_EXP_SEED = 12345

        env.seed(FIXED_EXP_SEED)
        env.reset()

        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()

        total_reward = 0
        done = False
        i = 0
        if scheme != 'pcaps' and scheme != 'cap_fifo' and scheme != 'cap_partition' and scheme != 'cap_decima' and scheme != 'green_hadoop':
            while not done:
                # print a single dot every 10 steps to indicate progress (all on same line)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                node, use_exec = agent.get_action(obs)
                obs, reward, done = env.step(node, use_exec)
                total_reward += reward
        elif scheme == 'pcaps':
            # refresh tensorflow completely
            tf.compat.v1.reset_default_graph() 
            # tf.compat.v1.set_random_seed(args.seed)
            
            # Set a fixed seed for TensorFlow to ensure reproducibility
            FIXED_TF_SEED = 42
            tf.compat.v1.set_random_seed(FIXED_TF_SEED)
            
            sess = tf.compat.v1.Session()
            # initialize scheduler
            agent = PCAPSAgent(
                sess, args.node_input_dim, args.job_input_dim,
                args.hid_dims, args.output_dim, args.max_depth,
                range(1, args.exec_cap + 1), carbon_dict)
            while not done:
                node, use_exec, cw = agent.get_action(obs)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                obs, reward, done = env.step(node, use_exec, carbon_aware = cw)
                total_reward += reward
        elif scheme == 'cap_decima':
            # refresh tensorflow completely
            tf.compat.v1.reset_default_graph() 
            tf.compat.v1.set_random_seed(args.seed)
            sess = tf.compat.v1.Session()
            agent = CarbonActorAgent(
                sess, args.node_input_dim, args.job_input_dim,
                args.hid_dims, args.output_dim, args.max_depth,
                range(1, args.exec_cap + 1), carbon_dict)
            while not done:
                node, use_exec, cw = agent.get_action(obs)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                obs, reward, done = env.step(node, use_exec, carbon_aware = cw)
                total_reward += reward
        elif scheme == 'cap_fifo' or scheme == 'cap_partition' or scheme == 'green_hadoop':
            while not done:
                # print a single dot every 10 steps to indicate progress (all on same line)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                node, use_exec, cw = agent.get_action(obs)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                obs, reward, done = env.step(node, use_exec, carbon_aware = cw)
                total_reward += reward

        all_total_reward[scheme].append(total_reward)
        
        job_dags = env.finished_job_dags

        job_durations = [
            job_dag.completion_time - job_dag.start_time
            for job_dag in job_dags
        ]
        
        # Calculate carbon for every power model using this one completed
        # schedule. The schedule is not rerun for different power models.
        default_carbon_value = carbon_dict[next(iter(carbon_dict))]
        schedule_end = int(env.wall_time.curr_time)

        print("")
        for model in power_models:
            pidle = model['pidle']
            pdyn = model['pdyn']

            total_dynamic_carbon_usage = 0.0
            total_idle_carbon_usage = 0.0

            # Dynamic emissions: task runtime * P_DYN_W * carbon intensity.
            for job_dag in env.finished_job_dags:
                for node in job_dag.nodes:
                    for task in node.tasks:
                        start = int(task.start_time)
                        finish = int(task.finish_time)

                        start_key = start - (start % 60000)
                        end_key = finish - (finish % 60000)

                        for key in range(start_key, end_key + 1, 60000):
                            carbon_value = carbon_dict.get(
                                key,
                                default_carbon_value
                            )

                            bucket_start = max(start, key)
                            bucket_end = min(finish, key + 60000)
                            duration = bucket_end - bucket_start

                            total_dynamic_carbon_usage += (
                                duration * pdyn * carbon_value
                            )

            # Always-on idle emissions: every provisioned executor consumes
            # P_IDLE_W from time 0 until the schedule ends.
            for key in range(0, schedule_end, 60000):
                bucket_end = min(key + 60000, schedule_end)
                duration = bucket_end - key
                carbon_value = carbon_dict.get(
                    key,
                    default_carbon_value
                )

                total_idle_carbon_usage += (
                    args.exec_cap
                    * pidle
                    * duration
                    * carbon_value
                )

            total_carbon_usage = (
                total_dynamic_carbon_usage
                + total_idle_carbon_usage
            )
            
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
                "pidle": pidle,
                "pdyn": pdyn,
                "dynamic_carbon_usage":
                    total_dynamic_carbon_usage,
                "idle_carbon_usage":
                    total_idle_carbon_usage,
                "total_carbon_usage":
                    total_carbon_usage,
            })

            print(
                f""
                f"Carbon usage — experiment {exp + 1}, "
                f"scheme {scheme}, model {model['name']}: "
                f"dynamic={total_dynamic_carbon_usage:.2f}, "
                f"idle={total_idle_carbon_usage:.2f}, "
                f"total={total_carbon_usage:.2f}"
            )
        
       
        # Add scheme data to results
        job_dags = env.finished_job_dags
        job_durations = [job_dag.completion_time - job_dag.start_time for job_dag in job_dags]
        
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

for model in power_models:
    model_name = model["name"]
    scheme_data = scheme_results_by_model[model_name]

    print(
        f"{model_name}: "
        f"{len(scheme_data)} entries"
    )

    if not scheme_data:
        print(
            f"Skipping {model_name}: "
            "no results were collected"
        )
        continue

    # Average the five experiments for each scheduler.
    averaged_scheme_data = {}

    for entry in scheme_data:
        scheme = entry[0]
        completion_time = entry[1]
        carbon_usage = entry[2]
        average_job_duration = entry[3]

        if scheme not in averaged_scheme_data:
            averaged_scheme_data[scheme] = {
                "completion_time": [],
                "carbon_usage": [],
                "job_duration": [],
            }

        averaged_scheme_data[scheme]["completion_time"].append(
            completion_time
        )
        averaged_scheme_data[scheme]["carbon_usage"].append(
            carbon_usage
        )
        averaged_scheme_data[scheme]["job_duration"].append(
            average_job_duration
        )

    averaged_data = []

    for scheme, values in averaged_scheme_data.items():
        averaged_data.append((
            scheme,
            np.mean(values["completion_time"]),
            np.mean(values["carbon_usage"]),
            np.mean(values["job_duration"]),
        ))

    print(
        f"Plotting {model_name} with "
        f"{len(averaged_data)} averaged scheduler results"
    )

    visualize_carbon_usage_aggregated(
        averaged_data,
        args.result_folder +
        "aggregated_carbon_usage_" +
        model_name +
        ".png"
    )