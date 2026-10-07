# efficiency_cap_fifo_agent.py adds carbon-efficiency tie-breaking to CAP FIFO.

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class EfficiencyCapFIFOAgent(BaseCarbonAgent):
    """CAP FIFO with carbon-efficiency ranking among near-arrival jobs."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 arrival_tolerance=60000, max_consecutive_waits=3):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=min(20, exec_cap),
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.arrival_tolerance = max(0, arrival_tolerance)
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0

    def _carbon_at(self, current_time):
        keys = sorted(self.carbon_schedule)
        previous = [key for key in keys if key <= current_time]
        if previous:
            return self.carbon_schedule[previous[-1]]
        return self.carbon_schedule[keys[0]]

    def _average_carbon(self, start_time, end_time):
        if end_time <= start_time:
            return self._carbon_at(start_time)
        boundaries = [start_time]
        boundaries.extend(
            key for key in sorted(self.carbon_schedule)
            if start_time < key < end_time
        )
        boundaries.append(end_time)
        total = 0.0
        duration = 0.0
        for begin, finish in zip(boundaries, boundaries[1:]):
            span = finish - begin
            total += span * self._carbon_at(begin)
            duration += span
        return total / max(duration, 1.0)

    def _node_work(self, node, exec_commit, moving_executors):
        remaining = self.remaining_tasks(node, exec_commit, moving_executors)
        if remaining <= 0:
            return 0.0, 0
        tasks = node.tasks[node.next_task_idx:node.next_task_idx + remaining]
        return float(sum(task.get_duration() for task in tasks)), remaining

    def _first_ready_node(self, job_dag, frontier_nodes):
        for node in job_dag.frontier_nodes:
            if node in frontier_nodes:
                return node
        for node in frontier_nodes:
            if node.job_dag == job_dag:
                return node
        return None

    def _job_efficiency(self, node, current_time, num_source_exec,
                        exec_commit, moving_executors):
        work, task_count = self._node_work(
            node, exec_commit, moving_executors)
        if work <= 0:
            return 0.0
        parallelism = max(1, min(task_count, num_source_exec, self.exec_cap_MAX))
        runtime = work / parallelism
        average_carbon = self._average_carbon(
            current_time, current_time + runtime)
        dynamic_cost = self.pdyn * work * average_carbon
        idle_cost = self.pidle * parallelism * runtime * average_carbon
        estimated_cost = dynamic_cost + idle_cost
        return work / max(estimated_cost, 1e-9)

    def _select_node(self, job_dags, frontier_nodes, source_job,
                     current_time, num_source_exec, exec_commit,
                     moving_executors):
        candidates = []
        for index, job_dag in enumerate(job_dags):
            if job_dag.name == 'dummy':
                continue
            node = self._first_ready_node(job_dag, frontier_nodes)
            if node is None:
                continue
            arrival = job_dag.start_time if job_dag.start_time is not None else 0
            candidates.append((index, arrival, job_dag, node))
        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[1], item[0]))
        head_arrival = candidates[0][1]
        eligible = [
            item for item in candidates
            if item[1] - head_arrival <= self.arrival_tolerance
        ]
        return max(
            eligible,
            key=lambda item: self._job_efficiency(
                item[3], current_time, num_source_exec,
                exec_commit, moving_executors),
        )[3]

    def _source_action(self, node, num_source_exec):
        available = max(int(self.last_available_cap), 0)
        use_exec = min(num_source_exec, available)
        if use_exec >= 1:
            self.consecutive_waits = 0
            return node, use_exec, False
        if self.consecutive_waits >= self.max_consecutive_waits:
            self.consecutive_waits = 0
            return node, num_source_exec, False
        self.consecutive_waits += 1
        return None, num_source_exec, True

    def get_action(self, obs):
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, exec_commit, \
            moving_executors, action_map, current_time = obs

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

        node = self._select_node(
            job_dags, frontier_nodes, source_job, current_time,
            num_source_exec, exec_commit, moving_executors)
        if node is None:
            if job_dags and num_source_exec > 0:
                if self.consecutive_waits >= self.max_consecutive_waits:
                    self.consecutive_waits = 0
                    return None, num_source_exec, False
                self.consecutive_waits += 1
                return None, num_source_exec, True
            return None, num_source_exec, False

        if source_job is not None and node.job_dag is source_job:
            return self._source_action(node, num_source_exec)

        available = max(int(self.last_available_cap), 0)
        remaining = self.remaining_tasks(node, exec_commit, moving_executors)
        use_exec = min(remaining, num_source_exec, available)
        if use_exec >= 1:
            self.consecutive_waits = 0
            return node, int(use_exec), False

        if self.consecutive_waits >= self.max_consecutive_waits:
            self.consecutive_waits = 0
            return node, min(remaining, num_source_exec), False
        self.consecutive_waits += 1
        return None, num_source_exec, True
