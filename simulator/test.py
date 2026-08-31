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


scheme_results = []

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
        
        executor_occupation = np.zeros(int(env.wall_time.curr_time))
        total_carbon_usage = 0
        default_carbon_value = carbon_dict[next(iter(carbon_dict))]

        print("")
        for job_dag in env.finished_job_dags:
            for node in job_dag.nodes:
                for task in node.tasks:
                    start_time = int(task.start_time)
                    finish_time = int(task.finish_time)
                    executor_occupation[
                    start_time:finish_time] += 1

                    start_key = start_time - (start_time % 60000)
                    end_key = finish_time - (finish_time % 60000)
                    for key in range(start_key, end_key + 1, 60000):
                        carbon_value = carbon_dict.get(key, default_carbon_value)
                        # Calculate overlap of this 100-second bucket with the task's time window
                        bucket_start = max(start_time, key)
                        bucket_end = min(finish_time, key + 60000)
                        duration_in_bucket = bucket_end - bucket_start
                        total_carbon_usage += duration_in_bucket * carbon_value
       
        # Add scheme data to results
        job_dags = env.finished_job_dags
        job_durations = [job_dag.completion_time - job_dag.start_time for job_dag in job_dags]

        scheme_results.append((
            scheme,  # Scheme name
            env.wall_time.curr_time,  # Total time (avg executors)
            total_carbon_usage,  # Total carbon usage
            np.mean(job_durations)  # Average job completion time
        ))
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
