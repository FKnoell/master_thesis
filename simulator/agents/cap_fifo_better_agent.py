# carbon_aware_fifo_agent.py implements CAP (Carbon-Aware Provisioning) on top of the default Spark FIFO behavior.


import numpy as np
from agents.agent import Agent
import math
from scipy.special import lambertw

class CarbonAgent(Agent):
    # statically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):


        # map for executor assignment
        self.exec_map = {}


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


    def count_allocated_executors(self, job_dags, exec_commit, moving_executors):
        allocated = sum(
            len(job_dag.executors)
            for job_dag in job_dags
            if job_dag.name != 'dummy'
        )


        allocated += sum(
            1
            for node in moving_executors.moving_executors.values()
            if node.job_dag.name != 'dummy'
        )


        # A commitment from the None pool represents an executor that has
        # not been attached to a job yet. Commitments from a job or node are
        # already included in that job's executor count above.
        for node, count in exec_commit.commit[None].items():
            if node is not None and node.job_dag.name != 'dummy':
                allocated += count


        return allocated
       


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


        # compute carbon thresholding
        controllable_k = self.exec_cap_MAX - self.B
        # solve for alpha (competitive ratio for k-search)
        alpha = 1 / (1 + lambertw( ( (self.L/self.U) - 1 ) / math.e ).real )
        thresholds = [ self.U*(1 - (1 - (1/alpha)) * (1 + (1/(alpha*controllable_k)))**(i-1) ) for i in range(1, controllable_k+1)]


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


        # Derive capacity from the environment instead of retaining a stale
        # estimate across task completions and executor movements.
        num_exec = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        available_exec = self.exec_cap - num_exec


        # the source job is finished or does not exist
        if available_exec <= 0:
            # we need to pause execution, so just return a null action
            return None, num_source_exec, True


        scheduled = False
        # first assign executor to the same job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    scaling = np.ceil( (self.exec_cap/self.exec_cap_MAX) * num_source_exec )
                    scaled = num_source_exec
                    if not np.isnan(scaling):
                        scaled = int(scaling)
                    diff = num_source_exec - scaled
                    candidate = num_source_exec -  max(0.8 - (self.L/self.U), 0.05)*diff
                    use_exec = num_source_exec
                    if not np.isnan(candidate):
                        use_exec = min(int(candidate), num_source_exec)
                    if use_exec < 1:
                        # return dummy task
                        return None, num_source_exec, True
                   
                    return node, min(
                        num_source_exec,
                        available_exec,
                        max(int(self.exec_cap / self.exec_cap_MAX), 1)), False


            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    scaling = np.ceil( (self.exec_cap/self.exec_cap_MAX) * num_source_exec )
                    scaled = num_source_exec
                    if not np.isnan(scaling):
                        scaled = int(scaling)
                    diff = num_source_exec - scaled
                    candidate = num_source_exec -  max(0.8 - (self.L/self.U), 0.05)*diff
                    use_exec = num_source_exec
                    if not np.isnan(candidate):
                        use_exec = min(int(candidate), num_source_exec)
                    if use_exec < 1:
                        # return dummy task
                        return None, num_source_exec, True
                   
                    return node, min(
                        num_source_exec,
                        available_exec,
                        max(int(self.exec_cap / self.exec_cap_MAX), 1)), False
       
        for job_dag in job_dags:
            if available_exec > 0:
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
                        num_source_exec,
                        available_exec)
                    use_exec = use_exec_init
                    if use_exec >= 1:
                        return node, use_exec, False
       
        return None, num_source_exec, False