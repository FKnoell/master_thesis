from agents.enhanced_cap_fifo_agent import EnhancedCapFIFOAgent


class CapPowerEnhancedFIFOAgent(EnhancedCapFIFOAgent):
    """Enhanced CAP FIFO with critical-path priorities and power-aware waits."""

    def __init__(self, exec_cap, carbon_schedule, pidle=0.0, pdyn=1.0,
                 horizon_slots=6, arrival_tolerance=180000,
                 max_backfill_work=120000.0, max_slowdown=1.05,
                 base_B=0.1, max_B=0.6):
        super().__init__(
            exec_cap=exec_cap,
            carbon_schedule=carbon_schedule,
            pidle=pidle,
            pdyn=pdyn,
            horizon_slots=horizon_slots,
            arrival_tolerance=arrival_tolerance,
            max_backfill_work=max_backfill_work,
            max_slowdown=max_slowdown,
            base_B=base_B,
            max_B=max_B,
        )
        self.last_wait_time = None

    def _critical_path_score(self, node, exec_commit, moving_executors,
                             memo=None):
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]

        own_work, _ = self._work(node, exec_commit, moving_executors)
        child_path = max(
            (
                self._critical_path_score(
                    child, exec_commit, moving_executors, memo)
                for child in node.child_nodes
            ),
            default=0.0,
        )
        descendant_work = sum(
            self._work(descendant, exec_commit, moving_executors)[0]
            for descendant in node.descendant_nodes
        )
        score = own_work + child_path + 0.1 * descendant_work
        memo[node] = score
        return score

    def _candidates(self, job_dags, frontier_nodes, current_time,
                    num_source_exec, exec_commit, moving_executors):
        job_indexes = {
            job_dag: index
            for index, job_dag in enumerate(job_dags)
            if job_dag.name != 'dummy'
        }
        candidates = []
        for node in frontier_nodes:
            index = job_indexes.get(node.job_dag)
            if index is None or self._work(
                    node, exec_commit, moving_executors)[0] <= 0:
                continue
            arrival = node.job_dag.start_time or 0
            candidates.append((index, arrival, node))

        memo = {}

        def priority(candidate):
            index, _, node = candidate
            job_age = max(
                0.0, current_time - (node.job_dag.start_time or 0))
            source_bonus = (
                0.05 if node.job_dag is self.current_source_job else 0.0
            )
            efficiency = self._efficiency(
                node, current_time, num_source_exec,
                exec_commit, moving_executors,
            )
            return (
                self._critical_path_score(
                    node, exec_commit, moving_executors, memo,
                ),
                source_bonus + job_age / 100000.0,
                efficiency,
                -index,
            )

        return sorted(candidates, key=priority, reverse=True)

    def _should_wait(self, node, job_dags, current_time, exec_commit,
                     moving_executors):
        if node is None or self.pdyn <= 0:
            return False

        current_carbon = self._carbon_at(current_time)
        future_slots = [
            key for key in sorted(self.carbon_schedule)
            if key > current_time
        ][:self.horizon_slots]
        future_slot = next(
            (key for key in future_slots
             if self.carbon_schedule[key] < current_carbon),
            None,
        )
        if future_slot is None:
            return False

        work, _ = self._work(node, exec_commit, moving_executors)
        if work <= 0:
            return False

        wait_duration = future_slot - current_time
        allocated = self.count_allocated_executors(
            job_dags, exec_commit, moving_executors)
        lower = min(self.B, self.exec_cap_MAX)
        baseline_cap = max(
            lower,
            min(self.exec_cap_MAX, max(1, allocated)),
        )
        baseline_runtime = work / baseline_cap
        fastest_runtime = work / max(1, self.exec_cap_MAX)
        if wait_duration + fastest_runtime > \
                baseline_runtime * self.max_slowdown:
            return False

        future_carbon = self.carbon_schedule[future_slot]
        execution_now = self.pdyn * work * current_carbon
        execution_later = self.pdyn * work * future_carbon
        waiting_cost = (
            self.pidle * self.exec_cap_MAX * wait_duration * current_carbon
        )
        return execution_later + waiting_cost < execution_now

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
        self.current_source_job = source_job
        self._update_lower_bound(job_dags, exec_commit, moving_executors)

        candidates = self._candidates(
            job_dags, frontier_nodes, current_time, num_source_exec,
            exec_commit, moving_executors,
        )
        node = candidates[0][2] if candidates else None
        can_wait = (
            self.consecutive_waits < self.max_consecutive_waits
            and current_time != self.last_wait_time
        )
        if can_wait and self._should_wait(
                node, job_dags, current_time, exec_commit,
                moving_executors):
            self.consecutive_waits += 1
            self.last_wait_time = current_time
            return None, num_source_exec, True

        return super().get_action(obs)
