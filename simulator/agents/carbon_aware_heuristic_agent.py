# carbon_aware_heuristic_agent.py implements CAP (Carbon-Aware Provisioning) on top of the heuristic Weighted Fair scheduler.

import numpy as np
from param import *
from agents.agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node
import math
from scipy.special import lambertw


class CarbonPartitionAgent(Agent):
    # dynamically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20,
                 pidle=0.0, pdyn=1.0):
        Agent.__init__(self, pidle=pidle, pdyn=pdyn)

        # carbon schedule
        self.carbon_schedule = carbon_schedule

        # carbon awareness
        self.L = 280
        self.U = 400
        self.B = exec_lower_bound
        self.thresholds = None 

        # executor limit set to each job
        self.exec_cap_MAX = exec_cap
        self.exec_cap = self.B

    def set_carbon_schedule(self, carbon_schedule):
        self.carbon_schedule = carbon_schedule

    def get_carbon_intensity(self, current_time):
        # round the current time to the nearest (previous) time in the keys
        keys = [key for key in self.carbon_schedule.keys() if key < current_time]
        if len(keys) > 0:
            current_CI_time = max(keys)
            current_carbon = self.carbon_schedule[current_CI_time]
            # get the keys for the 48 slots including and after current_CI_time
            future_keys = [key for key in self.carbon_schedule.keys() if key >= current_CI_time]
            L = 280
            U = 400
            if len(future_keys) > 48:
                future_keys = future_keys[:48]
                # get the carbon intensities for the next 48 slots
                future_carbon = [self.carbon_schedule[key] for key in future_keys]
                # get L and U, which are the minimum and maximum carbon intensities over the next 48 slots
                # L is the minimum carbon intensity over the next 48 slots
                # U is the maximum carbon intensity over the next 48 slots
                L = min(future_carbon)
                U = max(future_carbon)
            return current_carbon, L, U
        # else just return the first value
        else:
            current_carbon = self.carbon_schedule[list(self.carbon_schedule.keys())[0]]
            L = 280
            U = 400
            return current_carbon, L, U

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map, current_time = obs

        # get current carbon intensity
        current_carbon, L, U = self.get_carbon_intensity(current_time)
        self.L = L
        self.U = U

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([any(n.next_task_idx + \
            exec_commit.node_commit[n] + moving_executors.count(n) \
            < n.num_tasks for n in job_dag.nodes) \
            for job_dag in job_dags])
        
        # compute carbon thresholding
        controllable_k = self.exec_cap_MAX - self.B
        # solve for alpha (competitive ratio for k-search)
        alpha = 1 / (1 + lambertw( ( (self.L/self.U) - 1 ) / math.e ).real )
        thresholds = [ (U)*(1 - (1 - (1/alpha)) * (1 + (1/(controllable_k)))**(i) ) for i in range(1, controllable_k+1)]

        # find the first threshold that is greater than the current carbon intensity
        # since the thresholds are decreasing, the index of the first threshold that is less than 
        # the current carbon intensity is the number of allowable pods
        threshold_result = self.exec_cap_MAX
        for i, threshold in enumerate(thresholds):
            if threshold < current_carbon:
                threshold_result = self.B + i
                break

        # set the new exec cap
        self.exec_cap = threshold_result

        # compute the executor cap (per job)
        current_exec_cap = int(np.ceil(threshold_result / max(1, num_unfinished_jobs)))

        # sort out the exec_map, look at how many executors are currently up and running
        exec_map = {}
        num_exec = 0
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
            num_exec += exec_map[job_dag]
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            if node.job_dag.name != "dummy":
                exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j and n.job_dag.name != "dummy":
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # the source job is finished or does not exist
        if num_exec >= self.exec_cap:
            # we need to pause execution, so just return a null action
            if source_job is not None and source_job in list(exec_map):
                exec_map[source_job] = max(exec_map[source_job] - num_source_exec, 0)
            return None, num_source_exec, True

        scheduled = False
        # first assign executor to the same job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    scaled = int( np.ceil( (self.exec_cap/self.exec_cap_MAX) * num_source_exec ) )
                    diff = num_source_exec - scaled
                    candidate = num_source_exec -  max(0.8 - (self.L/self.U), 0.05)*diff
                    use_exec = num_source_exec
                    if not np.isnan(candidate):
                        use_exec = min(int(candidate), num_source_exec)
                    use_exec = max(use_exec, 1)

                    return node, min(num_source_exec, int(self.exec_cap/self.exec_cap_MAX)+1), False

            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    scaled = int( np.ceil( (self.exec_cap/self.exec_cap_MAX) * num_source_exec ) )
                    diff = num_source_exec - scaled
                    candidate = num_source_exec -  max(0.8 - (self.L/self.U), 0.05)*diff
                    use_exec = num_source_exec
                    if not np.isnan(candidate):
                        use_exec = min(int(candidate), num_source_exec)
                    use_exec = max(use_exec, 1)

                    return node, min(num_source_exec, int(self.exec_cap/self.exec_cap_MAX)+1), False
        
        for job_dag in job_dags:
            if exec_map[job_dag] < current_exec_cap:
                next_node = None
                # immediately scheduable node first
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break
                # then schedulable node in the job
                if next_node is None:
                    for node in frontier_nodes:
                        if node in job_dag.nodes:
                            next_node = node
                            break
                # node is selected, compute limit
                if next_node is not None:
                    use_exec_init = min(
                        node.num_tasks - node.next_task_idx - \
                        exec_commit.node_commit[node] - \
                        moving_executors.count(node),
                        num_source_exec)                    
                    use_exec = use_exec_init
                    return node, use_exec, False

        # there is more executors than tasks in the system
        return None, num_source_exec, False
