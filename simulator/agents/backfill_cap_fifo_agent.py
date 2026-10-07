# backfill_cap_fifo_agent.py adds power-aware FIFO backfilling to new_cap_fifo.

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class BackfillCapFIFOAgent(BaseCarbonAgent):
    """CAP FIFO that uses spare capacity for short, power-favorable jobs."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 max_backfill_work=120000.0, max_consecutive_waits=3):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=min(20, exec_cap),
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.max_backfill_work = max_backfill_work
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0

    def _carbon_at(self, current_time):
        keys = sorted(self.carbon_schedule)
        previous = [key for key in keys if key <= current_time]
        if previous:
            return self.carbon_schedule[previous[-1]]
        return self.carbon_schedule[keys[0]]

    def _remaining_work(self, node, exec_commit, moving_executors):
        remaining = self.remaining_tasks(node, exec_commit, moving_executors)
        if remaining <= 0:
            return 0.0
        tasks = node.tasks[node.next_task_idx:node.next_task_idx + remaining]
        return float(sum(task.get_duration() for task in tasks))

    def _first_ready_node(self, job_dag, frontier_nodes):
        for node in job_dag.frontier_nodes:
            if node in frontier_nodes:
                return node
        for node in frontier_nodes:
            if node.job_dag == job_dag:
                return node
        return None

    def _job_candidates(self, job_dags, frontier_nodes, exec_commit,
                        moving_executors):
        candidates = []
        for index, job_dag in enumerate(job_dags):
            if job_dag.name == 'dummy':
                continue
            node = self._first_ready_node(job_dag, frontier_nodes)
            if node is None:
                continue
            work = self._remaining_work(node, exec_commit, moving_executors)
            if work > 0:
                candidates.append((index, job_dag, node, work))
        return candidates

    def _backfill_is_favorable(self, work, spare, current_time):
        if work > self.max_backfill_work or spare <= 0:
            return False
        carbon = self._carbon_at(current_time)
        # Backfill when its dynamic cost is lower than leaving the spare
        # executors idle for an equivalent amount of work time. The factor
        # allows high idle-power models to fill more aggressively.
        dynamic_cost = self.pdyn * work * carbon
        idle_avoided = self.pidle * spare * work * carbon
        return dynamic_cost <= idle_avoided or self.pidle >= self.pdyn

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

        candidates = self._job_candidates(
            job_dags, frontier_nodes, exec_commit, moving_executors)
        if not candidates:
            if job_dags and num_source_exec > 0:
                if self.consecutive_waits >= self.max_consecutive_waits:
                    self.consecutive_waits = 0
                    return None, num_source_exec, False
                self.consecutive_waits += 1
                return None, num_source_exec, True
            return None, num_source_exec, False

        primary_index, primary_job, primary_node, primary_work = candidates[0]
        if source_job is not None:
            source_node = self._first_ready_node(source_job, frontier_nodes)
            if source_node is not None:
                return self._source_action(source_node, num_source_exec)

        available = max(int(self.last_available_cap), 0)
        primary_remaining = min(primary_work, num_source_exec, available)
        if primary_remaining >= 1:
            self.consecutive_waits = 0
            return primary_node, int(primary_remaining), False

        # The FIFO head cannot use the available pool. Use a short later job
        # only when it is power-favorable; otherwise wait for the head.
        spare = min(num_source_exec, max(available, 0))
        for index, job_dag, node, work in candidates[1:]:
            if index <= primary_index:
                continue
            if self._backfill_is_favorable(work, spare, current_time):
                self.consecutive_waits = 0
                return node, min(int(work), num_source_exec, spare), False

        if self.consecutive_waits >= self.max_consecutive_waits:
            self.consecutive_waits = 0
            return primary_node, min(int(primary_work), num_source_exec), False
        self.consecutive_waits += 1
        return None, num_source_exec, True
