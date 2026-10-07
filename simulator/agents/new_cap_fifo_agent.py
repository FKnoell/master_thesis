# new_cap_fifo_agent.py implements CAP (Carbon-Aware Provisioning) on top of the default Spark FIFO behavior.

import math

import numpy as np
from scipy.special import lambertw

from agents.agent import Agent


class CarbonAgent(Agent):
    # Preserve FIFO scheduling while deriving executor accounting from the
    # current environment observation instead of a local executor map.
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

        current_key = max(keys)
        current_carbon = self.carbon_schedule[current_key]
        future_keys = [
            key for key in self.carbon_schedule
            if key >= current_key
        ][:48]

        if len(future_keys) > 1:
            future_carbon = [
                self.carbon_schedule[key] for key in future_keys
            ]
            return current_carbon, min(future_carbon), max(future_carbon)

        return current_carbon, 280, 400

    @staticmethod
    def count_registered_executors(job_dags):
        return sum(
            len(job_dag.executors)
            for job_dag in job_dags
            if job_dag.name != "dummy"
        )

    @staticmethod
    def count_moving_executors(moving_executors):
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
        # Prefer the environment's node-level accounting representation.
        if hasattr(exec_commit, "node_commit"):
            return sum(
                count
                for node, count in exec_commit.node_commit.items()
                if (
                    node is not None
                    and node.job_dag is not None
                    and node.job_dag.name != "dummy"
                )
            )

        # Compatibility fallback for environments exposing only commit.
        total = 0
        for committed_nodes in exec_commit.commit.values():
            for node, count in committed_nodes.items():
                if (
                    node is not None
                    and node.job_dag is not None
                    and node.job_dag.name != "dummy"
                ):
                    total += count
        return total

    def count_allocated_executors(
        self,
        job_dags,
        exec_commit,
        moving_executors,
    ):
        return (
            self.count_registered_executors(job_dags)
            + self.count_moving_executors(moving_executors)
            + self.count_committed_executors(exec_commit)
        )

    def compute_exec_cap(self, current_carbon, L, U):
        controllable_k = int(self.exec_cap_MAX - self.B)

        if controllable_k <= 0:
            return int(self.B)
        if L <= 0 or U <= 0:
            return int(self.exec_cap_MAX)
        if L >= U:
            return int(
                self.exec_cap_MAX if current_carbon <= U else self.B
            )

        alpha = 1 / (1 + lambertw(((L / U) - 1) / math.e).real)
        self.thresholds = [
            U * (
                1
                - (1 - 1 / alpha)
                * (1 + 1 / (alpha * controllable_k)) ** (i - 1)
            )
            for i in range(1, controllable_k + 1)
        ]

        for i, threshold in enumerate(self.thresholds):
            if threshold < current_carbon:
                return int(self.B + i)

        return int(self.exec_cap_MAX)

    @staticmethod
    def remaining_tasks(node, exec_commit, moving_executors):
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
        if source_job is None or num_source_exec <= 0:
            return None

        # Preserve the original source-job priority.
        for node in source_job.frontier_nodes:
            if node in frontier_nodes:
                return self._source_action(node, num_source_exec)

        for node in frontier_nodes:
            if node.job_dag == source_job:
                return self._source_action(node, num_source_exec)

        return None

    def _source_action(self, node, num_source_exec):
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
        self.exec_cap = self.compute_exec_cap(current_carbon, L, U)

        # Fresh global snapshot. This replaces the old self.exec_map, but is
        # not used to alter FIFO scheduling decisions.
        num_allocated_exec = self.count_allocated_executors(
            job_dags,
            exec_commit,
            moving_executors,
        )
        self.last_allocated_executors = num_allocated_exec
        self.last_available_cap = self.exec_cap - num_allocated_exec

        source_action = self._source_job_action(
            source_job,
            num_source_exec,
            frontier_nodes,
        )
        if source_action is not None:
            return source_action

        # Preserve FIFO fallback behavior. The global count above is exposed
        # for provisioning/diagnostics, not used as a local per-job map.
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

            use_exec = min(
                self.remaining_tasks(
                    next_node,
                    exec_commit,
                    moving_executors,
                ),
                num_source_exec,
            )

            if use_exec >= 1:
                return next_node, int(use_exec), False

        return None, num_source_exec, False