# cap_fifo_backfill_agent.py implements CAP (Carbon-Aware Provisioning)
# on top of FIFO scheduling with backfilling.

import math

from scipy.special import lambertw

from agents.agent import Agent


class CarbonAgent(Agent):
    # FIFO with backfilling and carbon-aware provisioning.
    # Scheduling complexity:
    # O(num_jobs * num_nodes * num_executors)

    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)

        self.carbon_schedule = carbon_schedule

        # Carbon-awareness parameters
        self.L = 280
        self.U = 400
        self.B = exec_lower_bound

        # Executor cap
        self.exec_cap_MAX = exec_cap
        self.exec_cap = self.B

    def set_carbon_schedule(self, carbon_schedule):
        self.carbon_schedule = carbon_schedule

    def count_allocated_executors(
        self,
        job_dags,
        exec_commit,
        moving_executors,
    ):
        """Count executors currently allocated by the environment.

        Includes:
        1. Executors attached to real jobs.
        2. Executors currently moving to real jobs.
        3. Executors committed from the None pool to real jobs.

        Commitments from a job or node are not counted separately because
        they are already represented in the job's executor count.
        """
        allocated = sum(
            len(job_dag.executors)
            for job_dag in job_dags
            if job_dag.name != "dummy"
        )

        allocated += sum(
            1
            for node in moving_executors.moving_executors.values()
            if node.job_dag.name != "dummy"
        )

        # The None pool contains commitments from unassigned executors.
        for node, count in exec_commit.commit[None].items():
            if node is not None and node.job_dag.name != "dummy":
                allocated += count

        return allocated

    def get_carbon_intensity(self, current_time):
        """Return current carbon intensity and the future L/U bounds."""
        keys = [
            key
            for key in self.carbon_schedule.keys()
            if key < current_time
        ]

        if keys:
            current_ci_time = max(keys)
            current_carbon = self.carbon_schedule[current_ci_time]

            future_keys = [
                key
                for key in self.carbon_schedule.keys()
                if key >= current_ci_time
            ]

            L = 280
            U = 400

            if len(future_keys) > 48:
                future_keys = future_keys[:48]
                future_carbon = [
                    self.carbon_schedule[key]
                    for key in future_keys
                ]

                L = min(future_carbon)
                U = max(future_carbon)

            return current_carbon, L, U

        first_key = list(self.carbon_schedule.keys())[0]
        current_carbon = self.carbon_schedule[first_key]

        return current_carbon, 280, 400

    def compute_executor_cap(self, current_carbon):
        """Compute the carbon-aware global executor cap."""
        controllable_k = self.exec_cap_MAX - self.B

        if controllable_k <= 0:
            return self.B

        # Avoid division by zero or invalid threshold calculations.
        if self.U <= 0:
            return self.exec_cap_MAX

        carbon_ratio = self.L / self.U

        alpha = 1 / (
            1
            + lambertw(
                (carbon_ratio - 1) / math.e
            ).real
        )

        thresholds = [
            self.U
            * (
                1
                - (1 - (1 / alpha))
                * (
                    1
                    + (1 / (alpha * controllable_k))
                ) ** (i - 1)
            )
            for i in range(1, controllable_k + 1)
        ]

        for i, threshold in enumerate(thresholds):
            if threshold < current_carbon:
                return self.B + i

        return self.exec_cap_MAX

    @staticmethod
    def get_available_tasks(
        node,
        exec_commit,
        moving_executors,
    ):
        return (
            node.num_tasks
            - node.next_task_idx
            - exec_commit.node_commit[node]
            - moving_executors.count(node)
        )

    def get_action(self, obs):
        """
        FIFO scheduling with backfilling.

        1. Compute the carbon-aware global executor cap.
        2. Count currently allocated executors from the environment.
        3. Give the source job strict FIFO priority.
        4. Use remaining capacity to backfill later jobs.
        """
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

        # Update carbon information.
        current_carbon, L, U = self.get_carbon_intensity(current_time)
        self.L = L
        self.U = U

        # Compute the current global executor cap.
        self.exec_cap = self.compute_executor_cap(current_carbon)

        # Read the actual current executor allocation from the environment.
        num_exec = self.count_allocated_executors(
            job_dags,
            exec_commit,
            moving_executors,
        )

        available_exec = self.exec_cap - num_exec

        # The cap has been reached.
        if available_exec <= 0:
            return None, num_source_exec, True

        # Schedule the source job first.
        if source_job is not None:
            # Prefer immediately schedulable frontier nodes.
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    available_tasks = self.get_available_tasks(
                        node,
                        exec_commit,
                        moving_executors,
                    )

                    use_exec = min(
                        num_source_exec,
                        available_exec,
                        available_tasks,
                    )

                    if use_exec >= 1:
                        return node, use_exec, False

            # Fall back to any schedulable node belonging to source_job.
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    available_tasks = self.get_available_tasks(
                        node,
                        exec_commit,
                        moving_executors,
                    )

                    use_exec = min(
                        num_source_exec,
                        available_exec,
                        available_tasks,
                    )

                    if use_exec >= 1:
                        return node, use_exec, False

        # Backfill later jobs.
        # Only capacity unused by the source-job attempt is available.
        remaining_exec = available_exec

        if remaining_exec > 0:
            # job_dags is assumed to preserve FIFO order.
            for job_dag in job_dags:
                # The source job was already considered above.
                if source_job is not None and job_dag == source_job:
                    continue

                next_node = None

                # Prefer an immediately schedulable frontier node.
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break

                # Fall back to any schedulable node in this job.
                if next_node is None:
                    for node in frontier_nodes:
                        if node.job_dag == job_dag:
                            next_node = node
                            break

                if next_node is None:
                    continue

                available_tasks = self.get_available_tasks(
                    next_node,
                    exec_commit,
                    moving_executors,
                )

                use_exec = min(
                    num_source_exec,
                    remaining_exec,
                    available_tasks,
                )

                if use_exec >= 1:
                    return next_node, use_exec, False

        # No valid node was found.
        return None, num_source_exec, False