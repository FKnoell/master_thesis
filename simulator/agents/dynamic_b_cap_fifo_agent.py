# dynamic_b_cap_fifo_agent.py adds a workload- and power-dependent CAP lower bound.

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class DynamicBCapFIFOAgent(BaseCarbonAgent):
    """New CAP FIFO with a dynamic executor lower bound only."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 base_B=20, max_B=60, max_consecutive_waits=3):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=min(base_B, exec_cap),
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.base_B = max(1, base_B)
        self.max_B = min(max_B, exec_cap)
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0

    def compute_dynamic_B(self, job_dags, exec_commit, moving_executors):
        pending_tasks = 0
        for job_dag in job_dags:
            if job_dag.name == 'dummy':
                continue
            for node in job_dag.nodes:
                pending_tasks += self.remaining_tasks(
                    node, exec_commit, moving_executors)

        work_factor = min(1.0, pending_tasks / 1000.0)
        idle_penalty = 1.0 - 0.5 * max(0.0, min(1.0, self.pidle))
        dynamic_bonus = 0.25 * max(0.0, min(1.0, self.pdyn))
        dynamic_B = int(
            self.base_B * work_factor * (idle_penalty + dynamic_bonus)
        )
        return min(
            max(5, dynamic_B),
            self.max_B,
            self.exec_cap_MAX,
        )

    def _source_action(self, node, num_source_exec):
        available = max(int(self.last_available_cap), 0)
        use_exec = min(num_source_exec, available)
        if use_exec < 1:
            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                return node, num_source_exec, False
            self.consecutive_waits += 1
            return None, num_source_exec, True
        self.consecutive_waits = 0
        return node, use_exec, False

    def get_action(self, obs):
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, exec_commit, \
            moving_executors, action_map, current_time = obs

        self.B = self.compute_dynamic_B(
            job_dags, exec_commit, moving_executors)

        current_carbon, lower_carbon, upper_carbon = \
            self.get_carbon_intensity(current_time)
        self.L = lower_carbon
        self.U = upper_carbon
        self.exec_cap = self.compute_exec_cap(
            current_carbon, lower_carbon, upper_carbon)
        allocated = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        self.last_allocated_executors = allocated
        self.last_available_cap = self.exec_cap - allocated

        source_action = self._source_job_action(
            source_job, num_source_exec, frontier_nodes)
        if source_action is not None:
            return source_action

        available = max(int(self.last_available_cap), 0)
        for job_dag in job_dags:
            if job_dag.name == 'dummy':
                continue
            next_node = None
            for node in job_dag.frontier_nodes:
                if node in frontier_nodes:
                    next_node = node
                    break
            if next_node is None:
                for node in frontier_nodes:
                    if node.job_dag == job_dag:
                        next_node = node
                        break
            if next_node is None:
                continue

            remaining = self.remaining_tasks(
                next_node, exec_commit, moving_executors)
            use_exec = min(remaining, num_source_exec, available)
            if use_exec >= 1:
                self.consecutive_waits = 0
                return next_node, int(use_exec), False

            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                return next_node, min(remaining, num_source_exec), False

        if job_dags and num_source_exec > 0:
            self.consecutive_waits += 1
            return None, num_source_exec, True
        return None, num_source_exec, False
