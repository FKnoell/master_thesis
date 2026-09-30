# new_cap_decima_agent.py implements CAP (Carbon-Aware-Provisioning) on top of Decima.
#
# The original scheduling policy is preserved. Executor accounting is split into:
# - registered executors: currently available for execution;
# - allocated executors: registered + moving + committed, used for the global CAP provisioning limit.

import bisect
import math

import numpy as np
import tensorflow as tf
import tf_slim as slim
import tensorflow.compat.v1 as v1
from scipy.special import lambertw

from param import *
from utils import *
from tf_op import *
from msg_passing_path import *
from gcn import GraphCNN
from gsn import GraphSNN
from agents.agent import Agent


tf.compat.v1.disable_eager_execution()
v1.disable_v2_behavior()


class CarbonActorAgent(Agent):
    def __init__(
        self,
        sess,
        node_input_dim,
        job_input_dim,
        hid_dims,
        output_dim,
        max_depth,
        executor_levels,
        carbon_schedule,
        eps=1e-6,
        act_fn=leaky_relu,
        optimizer=tf.compat.v1.train.AdamOptimizer,
        scope="actor_agent",
        exec_lower_bound=20,
    ):
        super().__init__()

        self.sess = sess
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.carbon_schedule = carbon_schedule
        self.scope = scope

        self.L = 280
        self.U = 400
        self.B = exec_lower_bound
        self.thresholds = None
        self.exec_cap_MAX = executor_levels[-1] - 1
        self.exec_cap = self.B

        self.postman = Postman()

        self.node_inputs = tf.compat.v1.placeholder(
            tf.float32, [None, self.node_input_dim]
        )
        self.job_inputs = tf.compat.v1.placeholder(
            tf.float32, [None, self.job_input_dim]
        )

        self.gcn = GraphCNN(
            self.node_inputs,
            self.node_input_dim,
            self.hid_dims,
            self.output_dim,
            self.max_depth,
            self.act_fn,
            self.scope,
        )
        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim,
            self.hid_dims,
            self.output_dim,
            self.act_fn,
            self.scope,
        )

        self.node_valid_mask = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.job_valid_mask = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.dag_summ_backward_map = tf.compat.v1.placeholder(tf.float32, [None, None])

        self.node_act_probs, self.job_act_probs = self.actor_network(
            self.node_inputs,
            self.gcn.outputs,
            self.job_inputs,
            self.gsn.summaries[0],
            self.gsn.summaries[1],
            self.node_valid_mask,
            self.job_valid_mask,
            self.dag_summ_backward_map,
            self.act_fn,
        )

        node_logits = tf.math.log(self.node_act_probs)
        node_noise = tf.random.uniform(tf.shape(input=node_logits))
        self.node_acts = tf.argmax(
            input=node_logits - tf.math.log(-tf.math.log(node_noise)),
            axis=1,
        )

        job_logits = tf.math.log(self.job_act_probs)
        job_noise = tf.random.uniform(tf.shape(input=job_logits))
        self.job_acts = tf.argmax(
            input=job_logits - tf.math.log(-tf.math.log(job_noise)),
            axis=2,
        )

        self.node_act_vec = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.job_act_vec = tf.compat.v1.placeholder(tf.float32, [None, None, None])
        self.adv = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.entropy_weight = tf.compat.v1.placeholder(tf.float32, ())

        self.selected_node_prob = tf.reduce_sum(
            input_tensor=tf.multiply(self.node_act_probs, self.node_act_vec),
            axis=1,
            keepdims=True,
        )
        self.selected_job_prob = tf.reduce_sum(
            input_tensor=tf.reduce_sum(
                input_tensor=tf.multiply(self.job_act_probs, self.job_act_vec),
                axis=2,
            ),
            axis=1,
            keepdims=True,
        )
        self.adv_loss = tf.reduce_sum(
            input_tensor=tf.multiply(
                tf.math.log(self.selected_node_prob * self.selected_job_prob + self.eps),
                -self.adv,
            )
        )
        self.node_entropy = tf.reduce_sum(
            input_tensor=tf.multiply(
                self.node_act_probs,
                tf.math.log(self.node_act_probs + self.eps),
            )
        )
        self.prob_each_job = tf.reshape(
            tf.sparse.sparse_dense_matmul(
                self.gsn.summ_mats[0],
                tf.reshape(self.node_act_probs, [-1, 1]),
            ),
            [tf.shape(input=self.node_act_probs)[0], -1],
        )
        self.job_entropy = tf.reduce_sum(
            input_tensor=tf.multiply(
                self.prob_each_job,
                tf.reduce_sum(
                    input_tensor=tf.multiply(
                        self.job_act_probs,
                        tf.math.log(self.job_act_probs + self.eps),
                    ),
                    axis=2,
                ),
            )
        )
        self.entropy_loss = self.node_entropy + self.job_entropy
        self.entropy_loss /= (
            tf.math.log(tf.cast(tf.shape(input=self.node_act_probs)[1], tf.float32))
            + tf.math.log(float(len(self.executor_levels)))
        )
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        self.params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope,
        )
        self.input_params, self.set_params_op = self.define_params_op()
        self.act_gradients = tf.gradients(ys=self.act_loss, xs=self.params)
        self.lr_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)
        self.apply_grads = self.optimizer(self.lr_rate).apply_gradients(
            zip(self.act_gradients, self.params)
        )

        self.saver = tf.compat.v1.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(v1.global_variables_initializer())
        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(
        self,
        node_inputs,
        gcn_outputs,
        job_inputs,
        gsn_dag_summary,
        gsn_global_summary,
        node_valid_mask,
        job_valid_mask,
        gsn_summ_backward_map,
        act_fn,
    ):
        batch_size = tf.shape(input=node_valid_mask)[0]
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim]
        )
        job_inputs_reshape = tf.reshape(
            job_inputs, [batch_size, -1, self.job_input_dim]
        )
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim]
        )
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim]
        )
        gsn_summ_backward_map_extend = tf.tile(
            tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1]
        )
        gsn_dag_summ_extend = tf.matmul(
            gsn_summ_backward_map_extend,
            gsn_dag_summ_reshape,
        )
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim]
        )
        gsn_global_summ_extend_job = tf.tile(
            gsn_global_summ_reshape,
            [1, tf.shape(input=gsn_dag_summ_reshape)[1], 1],
        )
        gsn_global_summ_extend_node = tf.tile(
            gsn_global_summ_reshape,
            [1, tf.shape(input=gsn_dag_summ_extend)[1], 1],
        )

        with tf.compat.v1.variable_scope(self.scope):
            merge_node = tf.concat(
                [
                    node_inputs_reshape,
                    gcn_outputs_reshape,
                    gsn_dag_summ_extend,
                    gsn_global_summ_extend_node,
                ],
                axis=2,
            )
            node_hid_0 = slim.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = slim.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = slim.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = slim.fully_connected(node_hid_2, 1, activation_fn=None)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])
            node_outputs += (node_valid_mask - 1) * 10000.0
            node_outputs = tf.nn.softmax(node_outputs, axis=-1)

            merge_job = tf.concat(
                [
                    job_inputs_reshape,
                    gsn_dag_summ_reshape,
                    gsn_global_summ_extend_job,
                ],
                axis=2,
            )
            expanded_state = expand_act_on_state(
                merge_job,
                [level / 50.0 for level in self.executor_levels],
            )
            job_hid_0 = slim.fully_connected(expanded_state, 32, activation_fn=act_fn)
            job_hid_1 = slim.fully_connected(job_hid_0, 16, activation_fn=act_fn)
            job_hid_2 = slim.fully_connected(job_hid_1, 8, activation_fn=act_fn)
            job_outputs = slim.fully_connected(job_hid_2, 1, activation_fn=None)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])
            job_outputs += (job_valid_mask - 1) * 10000.0
            job_outputs = tf.reshape(
                job_outputs,
                [batch_size, -1, len(self.executor_levels)],
            )
            job_outputs = tf.nn.softmax(job_outputs, axis=-1)
            return node_outputs, job_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(
            self.apply_grads,
            feed_dict={
                i: d
                for i, d in zip(
                    self.act_gradients + [self.lr_rate],
                    gradients + [lr_rate],
                )
            },
        )

    def define_params_op(self):
        input_params = [
            tf.compat.v1.placeholder(tf.float32, shape=param.get_shape())
            for param in self.params
        ]
        set_params_op = [
            self.params[idx].assign(param)
            for idx, param in enumerate(input_params)
        ]
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run(
            [self.gsn.summaries],
            feed_dict={
                i: d
                for i, d in zip(
                    [self.node_inputs] + self.gsn.summ_mats,
                    [node_inputs] + summ_mats,
                )
            },
        )

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(
        self,
        node_inputs,
        job_inputs,
        node_valid_mask,
        job_valid_mask,
        gcn_mats,
        gcn_masks,
        summ_mats,
        running_dags_mat,
        dag_summ_backward_map,
        node_act_vec,
        job_act_vec,
        adv,
        entropy_weight,
    ):
        return self.sess.run(
            [self.act_gradients, [self.adv_loss, self.entropy_loss]],
            feed_dict={
                i: d
                for i, d in zip(
                    [self.node_inputs]
                    + [self.job_inputs]
                    + [self.node_valid_mask]
                    + [self.job_valid_mask]
                    + self.gcn.adj_mats
                    + self.gcn.masks
                    + self.gsn.summ_mats
                    + [self.dag_summ_backward_map]
                    + [self.node_act_vec]
                    + [self.job_act_vec]
                    + [self.adv]
                    + [self.entropy_weight],
                    [node_inputs]
                    + [job_inputs]
                    + [node_valid_mask]
                    + [job_valid_mask]
                    + gcn_mats
                    + gcn_masks
                    + [summ_mats, running_dags_mat]
                    + [dag_summ_backward_map]
                    + [node_act_vec]
                    + [job_act_vec]
                    + [adv]
                    + [entropy_weight],
                )
            },
        )

    def predict(
        self,
        node_inputs,
        job_inputs,
        node_valid_mask,
        job_valid_mask,
        gcn_mats,
        gcn_masks,
        summ_mats,
        running_dags_mat,
        dag_summ_backward_map,
    ):
        return self.sess.run(
            [self.node_act_probs, self.job_act_probs, self.node_acts, self.job_acts],
            feed_dict={
                i: d
                for i, d in zip(
                    [self.node_inputs]
                    + [self.job_inputs]
                    + [self.node_valid_mask]
                    + [self.job_valid_mask]
                    + self.gcn.adj_mats
                    + self.gcn.masks
                    + self.gsn.summ_mats
                    + [self.dag_summ_backward_map],
                    [node_inputs]
                    + [job_inputs]
                    + [node_valid_mask]
                    + [job_valid_mask]
                    + gcn_mats
                    + gcn_masks
                    + [summ_mats, running_dags_mat]
                    + [dag_summ_backward_map],
                )
            },
        )

    def set_params(self, input_params):
        self.sess.run(
            self.set_params_op,
            feed_dict={i: d for i, d in zip(self.input_params, input_params)},
        )

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

    def count_allocated_executors(self, job_dags, exec_commit, moving_executors):
        return (
            self.count_registered_executors(job_dags)
            + self.count_moving_executors(moving_executors)
            + self.count_committed_executors(exec_commit)
        )

    @staticmethod
    def count_unfinished_jobs(job_dags, exec_commit, moving_executors):
        return sum(
            1
            for job_dag in job_dags
            if job_dag.name != "dummy"
            and any(
                node.next_task_idx
                + exec_commit.node_commit[node]
                + moving_executors.count(node)
                < node.num_tasks
                for node in job_dag.nodes
            )
        )

    @staticmethod
    def remaining_tasks(node, exec_commit, moving_executors):
        return max(
            node.num_tasks
            - node.next_task_idx
            - exec_commit.node_commit[node]
            - moving_executors.count(node),
            0,
        )

    def compute_exec_cap(self, current_carbon, L, U):
        controllable_k = int(self.exec_cap_MAX - self.B)
        if controllable_k <= 0:
            return int(self.B)
        if U <= 0 or L <= 0:
            return int(self.exec_cap_MAX)
        if L >= U:
            return int(self.exec_cap_MAX if current_carbon <= U else self.B)

        alpha = 1 / (1 + lambertw(((L / U) - 1) / math.e).real)
        self.thresholds = [
            U
            * (
                1
                - (1 - (1 / alpha))
                * (1 + (1 / (alpha * controllable_k))) ** (i - 1)
            )
            for i in range(1, controllable_k + 1)
        ]

        for i, threshold in enumerate(self.thresholds):
            if threshold < current_carbon:
                return int(self.B + i)
        return int(self.exec_cap_MAX)

    def get_carbon_intensity(self, current_time):
        keys = [key for key in self.carbon_schedule if key < current_time]
        if not keys:
            first_key = next(iter(self.carbon_schedule))
            return self.carbon_schedule[first_key], 280, 400

        current_key = max(keys)
        current_carbon = self.carbon_schedule[current_key]
        future_keys = [key for key in self.carbon_schedule if key >= current_key][:48]
        if len(future_keys) > 1:
            future_carbon = [self.carbon_schedule[key] for key in future_keys]
            return current_carbon, min(future_carbon), max(future_carbon)
        return current_carbon, 280, 400

    def translate_state(self, obs):
        """Translate the observation to matrix form without a local exec_map."""
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

        total_num_nodes = int(
            sum(job_dag.num_nodes for job_dag in job_dags if job_dag.name != "dummy")
        )
        real_jobs = [job_dag for job_dag in job_dags if job_dag.name != "dummy"]
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(real_jobs), self.job_input_dim])

        registered_by_job = {
            job_dag: len(job_dag.executors)
            for job_dag in real_jobs
        }
        moving_by_job = {job_dag: 0 for job_dag in real_jobs}
        committed_by_job = {job_dag: 0 for job_dag in real_jobs}

        for node in moving_executors.moving_executors.values():
            if (
                node is not None
                and node.job_dag in moving_by_job
            ):
                moving_by_job[node.job_dag] += 1

        if hasattr(exec_commit, "node_commit"):
            for node, count in exec_commit.node_commit.items():
                if node is not None and node.job_dag in committed_by_job:
                    committed_by_job[node.job_dag] += count
        else:
            for committed_nodes in exec_commit.commit.values():
                for node, count in committed_nodes.items():
                    if node is not None and node.job_dag in committed_by_job:
                        committed_by_job[node.job_dag] += count

        allocated_by_job = {
            job_dag: (
                registered_by_job[job_dag]
                + moving_by_job[job_dag]
                + committed_by_job[job_dag]
            )
            for job_dag in real_jobs
        }

        job_idx_by_dag = {job_dag: idx for idx, job_dag in enumerate(real_jobs)}
        for job_dag, job_idx in job_idx_by_dag.items():
            job_inputs[job_idx, 0] = allocated_by_job[job_dag] / 20.0
            job_inputs[job_idx, 1] = 2 if job_dag is source_job else -2
            job_inputs[job_idx, 2] = num_source_exec / 20.0

        node_idx = 0
        for job_dag in real_jobs:
            job_idx = job_idx_by_dag[job_dag]
            for node in job_dag.nodes:
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]
                node_inputs[node_idx, 3] = (
                    node.num_tasks - node.next_task_idx
                ) * node.tasks[-1].duration / 100000.0
                node_inputs[node_idx, 4] = (
                    node.num_tasks - node.next_task_idx
                ) / 200.0
                node_idx += 1

        return (
            node_inputs,
            job_inputs,
            job_dags,
            source_job,
            num_source_exec,
            frontier_nodes,
            executor_limits,
            exec_commit,
            moving_executors,
            allocated_by_job,
            action_map,
        )

    def get_valid_masks(
        self,
        job_dags,
        frontier_nodes,
        source_job,
        num_source_exec,
        allocated_by_job,
        action_map,
    ):
        real_jobs = [job_dag for job_dag in job_dags if job_dag.name != "dummy"]
        job_valid_mask = np.zeros(
            [1, len(real_jobs) * len(self.executor_levels)]
        )
        job_valid = {}
        base = 0

        for job_dag in real_jobs:
            current_allocated = allocated_by_job[job_dag]
            if job_dag is source_job:
                least_exec_amount = current_allocated - num_source_exec + 1
            else:
                least_exec_amount = current_allocated + 1

            least_exec_amount = max(least_exec_amount, 1)
            exec_level_idx = bisect.bisect_left(
                self.executor_levels,
                least_exec_amount,
            )
            job_valid[job_dag] = exec_level_idx < len(self.executor_levels)

            for level_idx in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + level_idx] = 1

            base += len(self.executor_levels)

        total_num_nodes = int(
            sum(job_dag.num_nodes for job_dag in real_jobs)
        )
        node_valid_mask = np.zeros([1, total_num_nodes])
        for node in frontier_nodes:
            if node.job_dag in job_valid:
                if job_valid[node.job_dag]:
                    action_idx = action_map.inverse_map[node]
                    node_valid_mask[0, action_idx] = 1

        return node_valid_mask, job_valid_mask

    def invoke_model(self, obs):
        (
            node_inputs,
            job_inputs,
            job_dags,
            source_job,
            num_source_exec,
            frontier_nodes,
            executor_limits,
            exec_commit,
            moving_executors,
            allocated_by_job,
            action_map,
        ) = self.translate_state(obs)

        gcn_mats, gcn_masks, dag_summ_backward_map, running_dags_mat, job_dags_changed = (
            self.postman.get_msg_path(job_dags)
        )
        node_valid_mask, job_valid_mask = self.get_valid_masks(
            job_dags,
            frontier_nodes,
            source_job,
            num_source_exec,
            allocated_by_job,
            action_map,
        )
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)
        node_act_probs, job_act_probs, node_acts, job_acts = self.predict(
            node_inputs,
            job_inputs,
            node_valid_mask,
            job_valid_mask,
            gcn_mats,
            gcn_masks,
            summ_mats,
            running_dags_mat,
            dag_summ_backward_map,
        )

        return (
            node_acts,
            job_acts,
            node_act_probs,
            job_act_probs,
            node_inputs,
            job_inputs,
            node_valid_mask,
            job_valid_mask,
            gcn_mats,
            gcn_masks,
            summ_mats,
            running_dags_mat,
            dag_summ_backward_map,
            allocated_by_job,
            job_dags_changed,
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

        if not frontier_nodes:
            return None, num_source_exec, False

        (
            node_act,
            job_act,
            node_act_probs,
            job_act_probs,
            node_inputs,
            job_inputs,
            node_valid_mask,
            job_valid_mask,
            gcn_mats,
            gcn_masks,
            summ_mats,
            running_dags_mat,
            dag_summ_backward_map,
            allocated_by_job,
            job_dags_changed,
        ) = self.invoke_model(obs)

        num_allocated_exec = self.count_allocated_executors(
            job_dags,
            exec_commit,
            moving_executors,
        )
        _provisioning_headroom = max(self.exec_cap - num_allocated_exec, 0)

        if np.sum(node_valid_mask[0, :]) == 0:
            return None, num_source_exec, False

        assert node_valid_mask[0, node_act[0]] == 1
        node = action_map[node_act[0]]
        job_idx = job_dags.index(node.job_dag)

        assert job_valid_mask[0, job_act[0, job_idx] + len(self.executor_levels) * job_idx] == 1

        selected_level = self.executor_levels[job_act[0, job_idx]]
        current_job_allocated = allocated_by_job[node.job_dag]

        if node.job_dag is source_job:
            agent_exec_act = selected_level - current_job_allocated + num_source_exec
        else:
            agent_exec_act = selected_level - current_job_allocated

        use_exec = min(
            self.remaining_tasks(node, exec_commit, moving_executors),
            agent_exec_act,
            num_source_exec,
        )
        use_exec = max(int(use_exec), 0)

        if use_exec < 1:
            return None, num_source_exec, False

        return node, use_exec, False