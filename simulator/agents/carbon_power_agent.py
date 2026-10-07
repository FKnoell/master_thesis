# carbon_power_agent.py implements a fixed-cap scheduler that delays work when
# the predicted carbon savings exceed the power cost of waiting.

from agents.agent import Agent

class CarbonPowerAgent(Agent):
    """Capped, precedence-aware scheduling with power-aware deferral."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 max_consecutive_waits=3):
        Agent.__init__(self, pidle=pidle, pdyn=pdyn)
        self.exec_cap = exec_cap
        self.carbon_schedule = carbon_schedule
        self.exec_map = {}
        self.max_consecutive_waits = max_consecutive_waits
        self.consecutive_waits = 0
        self.last_wait_time = None

    def set_carbon_schedule(self, carbon_schedule):
        self.carbon_schedule = carbon_schedule

    def _carbon_at(self, current_time):
        keys = sorted(self.carbon_schedule)
        previous_keys = [key for key in keys if key <= current_time]
        if previous_keys:
            return self.carbon_schedule[previous_keys[-1]]
        return self.carbon_schedule[keys[0]]

    def _next_lower_carbon_slot(self, current_time, current_carbon):
        for key in sorted(self.carbon_schedule):
            if key > current_time and self.carbon_schedule[key] < current_carbon:
                return key, self.carbon_schedule[key]
        return None, None

    def _estimate_work(self, node, exec_commit, moving_executors):
        remaining = node.num_tasks - node.next_task_idx \
            - exec_commit.node_commit[node] - moving_executors.count(node)
        if remaining <= 0:
            return 0.0

        remaining_tasks = node.tasks[node.next_task_idx:node.next_task_idx + remaining]
        durations = [task.get_duration() for task in remaining_tasks]
        if not durations:
            return 0.0
        return float(sum(durations))

    def _critical_path_score(self, node, exec_commit, moving_executors,
                             memo=None):
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]

        own_work = self._estimate_work(node, exec_commit, moving_executors)
        child_path = 0.0
        for child in node.child_nodes:
            child_path = max(
                child_path,
                self._critical_path_score(
                    child, exec_commit, moving_executors, memo))

        # Favor bottlenecks, while retaining a smaller preference for total
        # remaining work so wide stages are not starved.
        score = own_work + child_path
        score += 0.1 * sum(
            self._estimate_work(descendant, exec_commit, moving_executors)
            for descendant in node.descendant_nodes
        )
        memo[node] = score
        return score

    def _candidate_nodes(self, job_dags, frontier_nodes, exec_commit,
                         moving_executors, current_time):
        candidates = [
            node for node in frontier_nodes
            if node.job_dag in job_dags
            and node.job_dag.name != 'dummy'
            and self._estimate_work(node, exec_commit, moving_executors) > 0
        ]
        memo = {}

        def priority(node):
            job_age = max(0.0, current_time - (node.job_dag.start_time or 0))
            source_bonus = 0.05 if node.job_dag is self.current_source_job \
                else 0.0
            return (
                self._critical_path_score(
                    node, exec_commit, moving_executors, memo),
                source_bonus + job_age / 100000.0,
            )

        return sorted(candidates, key=priority, reverse=True)

    @staticmethod
    def _count_allocated_executors(job_dags, exec_commit, moving_executors):
        allocated = sum(
            len(job_dag.executors)
            for job_dag in job_dags
            if job_dag.name != 'dummy'
        )
        allocated += sum(
            1
            for node in moving_executors.moving_executors.values()
            if node.job_dag.name != 'dummy'
        )
        for node, count in exec_commit.commit[None].items():
            if node is not None and node.job_dag.name != 'dummy':
                allocated += count
        return allocated

    def _should_wait(self, node, current_time, exec_commit, moving_executors):
        if node is None or self.pdyn <= 0:
            return False

        current_carbon = self._carbon_at(current_time)
        future_time, future_carbon = self._next_lower_carbon_slot(
            current_time, current_carbon)
        if future_time is None:
            return False

        work = self._estimate_work(node, exec_commit, moving_executors)
        if work <= 0:
            return False

        wait_duration = future_time - current_time
        execution_now = self.pdyn * work * current_carbon
        execution_later = self.pdyn * work * future_carbon
        waiting_cost = self.pidle * self.exec_cap * wait_duration * current_carbon
        return execution_later + waiting_cost < execution_now

    def get_action(self, obs):
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, exec_commit, \
            moving_executors, action_map, current_time = obs

        for job_dag in job_dags:
            self.exec_map.setdefault(job_dag, 0)
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]

        self.current_source_job = source_job
        candidates = self._candidate_nodes(
            job_dags, frontier_nodes, exec_commit, moving_executors,
            current_time)
        candidate = candidates[0] if candidates else None

        can_wait = self.consecutive_waits < self.max_consecutive_waits and \
            current_time != self.last_wait_time
        if can_wait and self._should_wait(
                candidate, current_time, exec_commit, moving_executors):
            self.consecutive_waits += 1
            self.last_wait_time = current_time
            return None, num_source_exec, True

        self.consecutive_waits = 0
        self.last_wait_time = None

        if candidate is not None:
            allocated = self._count_allocated_executors(
                job_dags, exec_commit, moving_executors)
            available = self.exec_cap - allocated
            remaining = candidate.num_tasks - candidate.next_task_idx \
                - exec_commit.node_commit[candidate] \
                - moving_executors.count(candidate)
            use_exec = min(remaining, available, num_source_exec)
            if use_exec > 0:
                self.exec_map[candidate.job_dag] += use_exec
                return candidate, use_exec, False

        if job_dags and num_source_exec > 0:
            return None, num_source_exec, True

        return None, num_source_exec, False