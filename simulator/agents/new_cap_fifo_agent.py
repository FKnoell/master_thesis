# new_cap_fifo_agent.py implements CAP (Carbon-Aware Provisioning) on top of the default Spark FIFO behavior.
#
# The original scheduling policy is preserved. Executor accounting is split into:
# - registered executors: executors attached to non-dummy jobs
# - moving executors: executors moving to real destination nodes
# - committed executors: active destination-node commitments

import math

import numpy as np
from scipy.special import lambertw

from agents.agent import Agent


class CarbonAgent(Agent):
    # statically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        super().__init__()

        self.carbon_schedule = carbon_schedule

        self.L = 280
        self.U = 400
        self.B = exec_lower_bound
        self.thresholds = None

        self.exec_cap_MAX = exec_cap
        self.exec_cap = self.B

    def set_carbon_schedule(self, carbon_schedule):
        self.carbon_schedule = carbon_schedule

    def get_carbon_intensity(self, current_time):
        keys = [key for key in self.carbon_schedule if key < current_time]

        if not keys:
            first_key = next(iter(self.carbon_schedule))
            return self.carbon_schedule[first_key], 280, 400

        current_time_key = max(keys)
        current_carbon = self.carbon_schedule[current_time_key]

        future_keys = [
            key for key in self.carbon_schedule
            if key >= current_time_key
        ][:48]

        if len(future_keys) > 1:
            future_carbon = [
                self.carbon_schedule[key] for key in future_keys
            ]
            return (
                current_carbon,
                min(future_carbon),
                max(future_carbon),
            )

        return current_carbon, 280, 400

    @staticmethod
    def count_registered_executors(job_dags):
        """Count executors currently registered with non-dummy jobs."""
        return sum(
            len(job_dag.executors)
            for job_dag in job_dags
            if job_dag.name != "dummy"
        )

    @staticmethod
    def count_moving_executors(moving_executors):
        """Count executors moving to real destination nodes."""
        return sum(
            1
            for node in moving_executors.moving_executors.values()
            if (
                node is not None
                and node.job_dag is not None
                and node.job_dag.name != "dummy"
            )
        )

    @staticmethod
    def count_committed_executors(exec_commit):
        """Count active commitments for non-dummy destination nodes."""
        return sum(
            count
            for node, count in exec_commit.node_commit.items()
            if (
                node is not None
                and node.job_dag is not None
                and node.job_dag.name != "dummy"
            )
        )

    def count_allocated_executors(
        self,
        job_dags,
        exec_commit,
        moving_executors,
    ):
        """Return the global allocated-executor count from the observation."""
        return (
            self.count_registered_executors(job_dags)
            + self.count_moving_executors(moving_executors)
            + self.count_committed_executors(exec_commit)
        )

    def compute_exec_cap(self, current_carbon, L, U):
        """Compute the carbon-aware global provisioning cap."""
        controllable_k = self.exec_cap_MAX - self.B

        if controllable_k <= 0:
            return self.B

        # Avoid a division-by-zero or invalid threshold calculation when the
        # forecast has no range.
        if U <= 0 or L <= 0:
            return self.exec_cap_MAX

        if L >= U:
            return self.exec_cap_MAX if current_carbon <= U else self.B

        alpha = 1 / (
            1 + lambertw(((L / U) - 1) / math.e).real
        )

        thresholds = [
            U * (
                1
                - (1 - (1 / alpha))
                * (1 + (1 / (alpha * controllable_k))) ** (i - 1)
            )
            for i in range(1, controllable_k + 1)
        ]
        self.thresholds = thresholds

        for i, threshold in enumerate(thresholds):
            if threshold < current_carbon:
                return self.B + i

        return self.exec_cap_MAX

    @staticmethod
    def _remaining_tasks(node, exec_commit, moving_executors):
        return max(
            node.num_tasks
            - node.next_task_idx
            - exec_commit.node_commit[node]
            - moving_executors.count(node),
            0,
        )

    def _source_job_action(
        self,
        source_job,
        num_source_exec,
        frontier_nodes,
    ):
        """Preserve the original FIFO preference for the source job."""
        if source_job is None or num_source_exec <= 0:
            return None

        candidate_nodes = [
            node for node in source_job.frontier_nodes
            if node in frontier_nodes
        ]

        candidate_nodes.extend(
            node for node in frontier_nodes
            if node.job_dag == source_job and node not in candidate_nodes
        )

        for node in candidate_nodes:
            scaling = np.ceil(
                (self.exec_cap / self.exec_cap_MAX) * num_source_exec
            )
            scaled = int(scaling) if not np.isnan(scaling) else num_source_exec
            diff = num_source_exec - scaled
            candidate = num_source_exec - max(
                0.8 - (self.L / self.U),
                0.05,
            ) * diff

            use_exec = (
                min(int(candidate), num_source_exec)
                if not np.isnan(candidate)
                else num_source_exec
            )

            if use_exec < 1:
                return None, num_source_exec, True

            return (
                node,
                min(
                    num_source_exec,
                    max(int(self.exec_cap / self.exec_cap_MAX), 1),
                ),
                False,
            )

        return None

    def get_action(self, obs):
        (
            job_dags,
            source_job,
            num_source_exec,
            frontier_nodes,
            executor_limits,
            exec_commit,
            moving_executors,
            action_map,
            current_time,
        ) = obs

        current_carbon, L, U = self.get_carbon_intensity(current_time)
        self.L = L
        self.U = U

        # This is the only executor state used by the agent. It is derived
        # from the current environment observation on every decision.
        num_allocated_exec = self.count_allocated_executors(
            job_dags,
            exec_commit,
            moving_executors,
        )

        # Compute the current global CAP from carbon conditions.
        self.exec_cap = self.compute_exec_cap(current_carbon, L, U)
        available_exec = self.exec_cap - num_allocated_exec

        # A full CAP blocks provisioning, not scheduling of executors already
        # available to source_job. Preserve FIFO source-job scheduling.
        source_action = self._source_job_action(
            source_job,
            num_source_exec,
            frontier_nodes,
        )
        if source_action is not None:
            return source_action

        # Preserve FIFO fallback scheduling.
        # The environment observation remains the global source of truth.
        for job_dag in job_dags:
            if job_dag.name == "dummy":
                continue

            next_node = None

            for node in job_dag.frontier_nodes:
                if node in frontier_nodes:
                    next_node = node
                    break

            if next_node is None:
                for node in frontier_nodes:
                    if node in job_dag.nodes:
                        next_node = node
                        break

            if next_node is None:
                continue

            remaining_tasks = self._remaining_tasks(
                next_node,
                exec_commit,
                moving_executors,
            )

            use_exec = min(
                remaining_tasks,
                num_source_exec,
            )

            if use_exec >= 1:
                return next_node, int(use_exec), False

        return None, num_source_exec, False