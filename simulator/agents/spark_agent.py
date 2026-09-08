# spark_agent.py implements the default Spark FIFO behavior.


import numpy as np
from agents.agent import Agent



class SparkAgent(Agent):
    # statically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap):
        Agent.__init__(self)


        # executor limit set to each job
        self.exec_cap = exec_cap


    def count_job_executors(self, job_dag, exec_commit, moving_executors):
        """Count executors allocated to a specific job from environment state"""
        allocated = len(job_dag.executors)


        allocated += sum(
            1
            for node in moving_executors.moving_executors.values()
            if node.job_dag == job_dag
        )


        # A commitment from the None pool represents an executor that has
        # not been attached to a job yet. Commitments from a job or node are
        # already included in that job's executor count above.
        for node, count in exec_commit.commit[None].items():
            if node is not None and node.job_dag == job_dag:
                allocated += count


        return allocated


    def get_action(self, obs):


        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map, _ = obs


        scheduled = False
        # first assign executor to the same job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    return node, num_source_exec
            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    return node, num_source_exec


        # the source job is finished or does not exist
        for job_dag in job_dags:
            job_exec_count = self.count_job_executors(job_dag, exec_commit, moving_executors)
            
            if job_exec_count < self.exec_cap:
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
                    use_exec = min(
                        node.num_tasks - node.next_task_idx - \
                        exec_commit.node_commit[node] - \
                        moving_executors.count(node),
                        self.exec_cap - job_exec_count,
                        num_source_exec)
                    use_exec = max(1, use_exec)
                    return next_node, use_exec


        # there is more executors than tasks in the system
        return None, num_source_exec