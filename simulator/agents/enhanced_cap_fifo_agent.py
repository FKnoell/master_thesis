# enhanced_cap_fifo_agent.py combines the FIFO CAP improvements.

import math

from agents.new_cap_fifo_agent import CarbonAgent as BaseCarbonAgent


class EnhancedCapFIFOAgent(BaseCarbonAgent):
    """FIFO CAP with dynamic B, lookahead, efficiency ties, and backfill."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 horizon_slots=6, arrival_tolerance=180000,
                 max_backfill_work=120000.0, max_slowdown=1.05,
                 base_B=0.1, max_B=0.6):
        if not 0 <= base_B <= 1 or not 0 <= max_B <= 1:
            raise ValueError('base_B and max_B must be fractions from 0 to 1')
        base_B = min(exec_cap, max(1, math.ceil(exec_cap * base_B)))
        max_B = min(exec_cap, max(1, math.ceil(exec_cap * max_B)))
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            exec_lower_bound=base_B,
        )
        self.pidle = pidle
        self.pdyn = pdyn
        self.max_B = max_B
        self.base_B = base_B
        self.horizon_slots = max(1, horizon_slots)
        self.arrival_tolerance = max(0, arrival_tolerance)
        self.max_backfill_work = max_backfill_work
        self.max_slowdown = max(1.0, max_slowdown)
        self.max_consecutive_waits = 11 - (pidle * 10)
        self.consecutive_waits = 0

    def _carbon_at(self, timestamp):
        keys = sorted(self.carbon_schedule)
        previous = [key for key in keys if key <= timestamp]
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

    def _work(self, node, exec_commit, moving_executors):
        remaining = self.remaining_tasks(node, exec_commit, moving_executors)
        if remaining <= 0:
            return 0.0, 0
        tasks = node.tasks[node.next_task_idx:node.next_task_idx + remaining]
        return float(sum(task.get_duration() for task in tasks)), remaining

    def _ready_node(self, job_dag, frontier_nodes):
        for node in job_dag.frontier_nodes:
            if node in frontier_nodes:
                return node
        for node in frontier_nodes:
            if node.job_dag == job_dag:
                return node
        return None

    def _pending_tasks(self, job_dags, exec_commit, moving_executors):
        return sum(
            self.remaining_tasks(node, exec_commit, moving_executors)
            for job_dag in job_dags
            if job_dag.name != 'dummy'
            for node in job_dag.nodes
        )

    def _update_lower_bound(self, job_dags, exec_commit, moving_executors):
        pending = self._pending_tasks(
            job_dags, exec_commit, moving_executors)
        work_factor = min(1.0, pending / 1000.0)
        idle_penalty = 1.0 - 0.5 * max(0.0, min(1.0, self.pidle))
        dynamic_bonus = 0.25 * max(0.0, min(1.0, self.pdyn))
        dynamic_B = int(
            self.base_B * work_factor * (idle_penalty + dynamic_bonus)
        )
        self.B = min(
            self.max_B,
            max(min(5, self.max_B), dynamic_B),
        )

    def _efficiency(self, node, current_time, num_source_exec,
                    exec_commit, moving_executors):
        work, tasks = self._work(node, exec_commit, moving_executors)
        if work <= 0:
            return 0.0
        parallelism = max(1, min(tasks, num_source_exec, self.exec_cap_MAX))
        runtime = work / parallelism
        carbon = self._average_carbon(current_time, current_time + runtime)
        cost = carbon * (self.pdyn * work + self.pidle * parallelism * runtime)
        return work / max(cost, 1e-9)

    def _candidates(self, job_dags, frontier_nodes, current_time,
                    num_source_exec, exec_commit, moving_executors):
        candidates = []
        for index, job_dag in enumerate(job_dags):
            if job_dag.name == 'dummy':
                continue
            node = self._ready_node(job_dag, frontier_nodes)
            if node is None or self._work(
                    node, exec_commit, moving_executors)[0] <= 0:
                continue
            arrival = job_dag.start_time if job_dag.start_time is not None else 0
            candidates.append((index, arrival, node))
        if not candidates:
            return []
        candidates.sort(key=lambda item: (item[1], item[0]))
        first_arrival = candidates[0][1]
        eligible = [
            item for item in candidates
            if item[1] - first_arrival <= self.arrival_tolerance
        ]
        return sorted(
            eligible,
            key=lambda item: self._efficiency(
                item[2], current_time, num_source_exec,
                exec_commit, moving_executors),
            reverse=True,
        )

    def _choose_cap(self, node, current_time, allocated, exec_commit,
                    moving_executors):
        if node is None:
            return self.exec_cap_MAX
        work, _ = self._work(node, exec_commit, moving_executors)
        if work <= 0:
            return self.exec_cap_MAX
        lower = min(self.B, self.exec_cap_MAX)
        baseline = max(lower, min(self.exec_cap_MAX, max(1, allocated)))
        baseline_runtime = work / baseline
        current = self._carbon_at(current_time)
        best = None
        future = [
            key for key in sorted(self.carbon_schedule)
            if key > current_time
        ][:self.horizon_slots]
        for cap in range(lower, self.exec_cap_MAX + 1):
            runtime = work / cap
            for delay in [0] + [key - current_time for key in future]:
                if delay + runtime > baseline_runtime * self.max_slowdown:
                    continue
                carbon = self._average_carbon(
                    current_time + delay, current_time + delay + runtime)
                cost = self.pdyn * work * carbon
                cost += self.pidle * cap * runtime * carbon
                cost += self.pidle * allocated * delay * current
                candidate = (cost, delay + runtime, -cap, cap)
                if best is None or candidate < best:
                    best = candidate
        return best[3] if best else max(lower, min(self.exec_cap_MAX, allocated))

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
        self._update_lower_bound(job_dags, exec_commit, moving_executors)
        current, lower, upper = self.get_carbon_intensity(current_time)
        self.L = lower
        self.U = upper
        candidates = self._candidates(
            job_dags, frontier_nodes, current_time, num_source_exec,
            exec_commit, moving_executors)
        node = candidates[0][2] if candidates else None
        self.exec_cap = self._choose_cap(
            node, current_time,
            self.count_allocated_executors(
                job_dags, exec_commit, moving_executors),
            exec_commit, moving_executors)
        allocated = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        self.last_available_cap = self.exec_cap - allocated

        if node is not None and source_job is not None \
                and node.job_dag is source_job:
            return self._source_action(node, num_source_exec)

        available = max(int(self.last_available_cap), 0)
        if node is not None:
            remaining = self.remaining_tasks(
                node, exec_commit, moving_executors)
            use_exec = min(remaining, num_source_exec, available)
            if use_exec >= 1:
                self.consecutive_waits = 0
                return node, int(use_exec), False

        # Backfill a short later job only when spare capacity avoids more
        # idle carbon than the dynamic work costs.
        if available > 0 and candidates:
            current_carbon = self._carbon_at(current_time)
            for _, _, backfill_node in candidates[1:]:
                work, _ = self._work(
                    backfill_node, exec_commit, moving_executors)
                if work <= self.max_backfill_work and (
                    self.pidle * available >= self.pdyn * work
                    or self.pidle >= self.pdyn
                ):
                    return backfill_node, min(
                        int(work), num_source_exec, available), False

        if job_dags and num_source_exec > 0:
            if self.consecutive_waits >= self.max_consecutive_waits:
                self.consecutive_waits = 0
                if node is not None:
                    return node, min(
                        self.remaining_tasks(
                            node, exec_commit, moving_executors),
                        num_source_exec,
                    ), False
            self.consecutive_waits += 1
            return None, num_source_exec, True
        return None, num_source_exec, False
