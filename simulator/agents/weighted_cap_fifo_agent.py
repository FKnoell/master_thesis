# weighted_cap_fifo_agent.py adds power-weighted CAP provisioning to new_cap_fifo_agent.

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class CarbonPowerAgent(BaseCarbonAgent):
    """CAP FIFO with utilization-aware power-adjusted thresholds."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 rho=0.7, exec_lower_bound=20, max_consecutive_waits=3):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=exec_lower_bound,
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.rho = max(0.0, rho)
        self.utilization = 0.0
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0
        self.last_wait_time = None

    def compute_exec_cap(self, current_carbon, lower_carbon,
                         upper_carbon):
        base_cap = super().compute_exec_cap(
            current_carbon, lower_carbon, upper_carbon)

        carbon_range = max(upper_carbon - lower_carbon, 1e-9)
        carbon_pressure = max(0.0, min(
            1.0,
            (current_carbon - lower_carbon) / carbon_range,
        ))
        power_total = max(self.pidle + self.rho * self.pdyn, 1e-9)
        idle_share = max(0.0, min(1.0, self.pidle / power_total))
        power_reduction = idle_share * carbon_pressure

        # Let high idle power lower the effective CAP floor as well as the
        # discretionary range. This keeps rho relevant when normal CAP has
        # already collapsed to its fixed lower bound.
        effective_floor = max(
            1,
            int(self.B * (1.0 - power_reduction)),
        )
        adjustment = 1.0 - power_reduction * (1.0 - self.utilization)
        adjusted_cap = effective_floor + int(
            (base_cap - effective_floor) * adjustment
        )
        adjusted_cap = max(effective_floor, min(adjusted_cap, base_cap))

        self.last_base_cap = base_cap
        self.last_carbon_pressure = carbon_pressure
        self.last_idle_share = idle_share
        self.last_effective_floor = effective_floor
        self.last_adjusted_cap = adjusted_cap
        return max(1, min(adjusted_cap, self.exec_cap_MAX))

    def _update_utilization(self, obs):
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, exec_commit, \
            moving_executors, action_map, current_time = obs

        allocated = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        running = sum(
            1
            for job_dag in job_dags
            if job_dag.name != 'dummy'
            for executor in job_dag.executors
            if executor.task is not None
            and executor.task.finish_time > current_time
        )
        alpha = 0.3
        new_util = running / max(allocated, 1)
        self.utilization = alpha * new_util + (1 - alpha) * self.utilization

    def _source_action(self, node, num_source_exec):
        available = max(int(self.last_available_cap), 0)
        use_exec = min(num_source_exec, available)
        if use_exec < 1:
            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                self.last_wait_time = None
                return node, num_source_exec, False
            self.consecutive_waits += 1
            return None, num_source_exec, True
        self.consecutive_waits = 0
        self.last_wait_time = None
        return node, use_exec, False

    def _wait_for_capacity(self, num_source_exec, current_time):
        if (self.consecutive_waits >= self.max_consecutive_waits
                or current_time == self.last_wait_time):
            self.consecutive_waits = 0
            self.last_wait_time = None
            return False
        self.consecutive_waits += 1
        self.last_wait_time = current_time
        return True

    def get_action(self, obs):
        self._update_utilization(obs)
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

            use_exec = min(
                self.remaining_tasks(
                    next_node, exec_commit, moving_executors),
                num_source_exec,
                available,
            )
            if use_exec >= 1:
                self.consecutive_waits = 0
                self.last_wait_time = None
                return next_node, int(use_exec), False

            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                self.last_wait_time = None
                return next_node, min(
                    self.remaining_tasks(
                        next_node, exec_commit, moving_executors),
                    num_source_exec,
                ), False

        if job_dags and num_source_exec > 0:
            if self._wait_for_capacity(num_source_exec, current_time):
                return None, num_source_exec, True
            for job_dag in job_dags:
                if job_dag.name == 'dummy':
                    continue
                for next_node in frontier_nodes:
                    if next_node.job_dag == job_dag:
                        return next_node, min(
                            self.remaining_tasks(
                                next_node, exec_commit, moving_executors),
                            num_source_exec,
                        ), False
            return None, num_source_exec, True
        return None, num_source_exec, False
