# lookahead_cap_fifo_agent.py implements short-horizon power-aware CAP FIFO.

import math

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class LookaheadCapFIFOAgent(BaseCarbonAgent):
    """CAP FIFO that chooses capacity by explicit short-horizon cost."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 horizon_slots=6, max_slowdown=1.25,
                 max_consecutive_waits=3):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=min(20, exec_cap),
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.horizon_slots = max(1, horizon_slots)
        self.max_slowdown = max(1.0, max_slowdown)
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0

    def _carbon_at(self, timestamp):
        keys = sorted(self.carbon_schedule)
        previous = [key for key in keys if key <= timestamp]
        if previous:
            return self.carbon_schedule[previous[-1]]
        return self.carbon_schedule[keys[0]]

    def _future_slots(self, current_time):
        keys = [
            key for key in sorted(self.carbon_schedule)
            if key > current_time
        ]
        return keys[:self.horizon_slots]

    def _average_carbon(self, start_time, end_time):
        if end_time <= start_time:
            return self._carbon_at(start_time)

        boundaries = [start_time]
        boundaries.extend(
            key for key in sorted(self.carbon_schedule)
            if start_time < key < end_time
        )
        boundaries.append(end_time)
        weighted = 0.0
        duration = 0.0
        for begin, finish in zip(boundaries, boundaries[1:]):
            span = finish - begin
            weighted += span * self._carbon_at(begin)
            duration += span
        return weighted / max(duration, 1.0)

    def _estimate_work(self, node, exec_commit, moving_executors):
        remaining = self.remaining_tasks(
            node, exec_commit, moving_executors)
        if remaining <= 0:
            return 0.0
        tasks = node.tasks[node.next_task_idx:node.next_task_idx + remaining]
        return float(sum(task.get_duration() for task in tasks))

    def _choose_cap(self, node, current_time, allocated, exec_commit,
                    moving_executors):
        if node is None:
            return self.exec_cap_MAX

        work = self._estimate_work(node, exec_commit, moving_executors)
        if work <= 0:
            return self.exec_cap_MAX

        lower = min(self.B, self.exec_cap_MAX)
        baseline_cap = max(lower, min(self.exec_cap_MAX, max(1, allocated)))
        baseline_runtime = work / max(baseline_cap, 1)
        max_completion = baseline_runtime * self.max_slowdown
        current_carbon = self._carbon_at(current_time)
        future_slots = self._future_slots(current_time)
        delays = [0] + [key - current_time for key in future_slots]
        candidates = []

        for cap in range(lower, self.exec_cap_MAX + 1):
            runtime = work / max(cap, 1)
            for delay in delays:
                if delay + runtime > max_completion:
                    continue
                start = current_time + delay
                finish = start + runtime
                execution_carbon = self._average_carbon(start, finish)
                dynamic_cost = self.pdyn * work * execution_carbon
                executor_idle_cost = self.pidle * cap * runtime * execution_carbon
                waiting_idle_cost = (
                    self.pidle * allocated * delay * current_carbon
                )
                total_cost = (
                    dynamic_cost + executor_idle_cost + waiting_idle_cost
                )
                candidates.append((total_cost, delay + runtime, -cap, cap))

        if not candidates:
            return min(self.exec_cap_MAX, max(lower, allocated))
        return min(candidates)[3]

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

        candidate = None
        if source_job is not None:
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    candidate = node
                    break
            if candidate is None:
                for node in frontier_nodes:
                    if node.job_dag == source_job:
                        candidate = node
                        break

        if candidate is None:
            for job_dag in job_dags:
                if job_dag.name == 'dummy':
                    continue
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        candidate = node
                        break
                if candidate is None:
                    for node in frontier_nodes:
                        if node.job_dag == job_dag:
                            candidate = node
                            break
                if candidate is not None:
                    break

        allocated = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        self.exec_cap = self._choose_cap(
            candidate, current_time, allocated, exec_commit, moving_executors)
        self.last_allocated_executors = allocated
        self.last_available_cap = self.exec_cap - allocated

        if source_job is not None and candidate is not None:
            return self._source_action(candidate, num_source_exec)

        if candidate is not None:
            available = max(int(self.last_available_cap), 0)
            remaining = self.remaining_tasks(
                candidate, exec_commit, moving_executors)
            use_exec = min(remaining, num_source_exec, available)
            if use_exec >= 1:
                self.consecutive_waits = 0
                return candidate, int(use_exec), False

        if job_dags and num_source_exec > 0:
            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                if candidate is not None:
                    return candidate, min(
                        self.remaining_tasks(
                            candidate, exec_commit, moving_executors),
                        num_source_exec,
                    ), False
            self.consecutive_waits += 1
            return None, num_source_exec, True
        return None, num_source_exec, False
