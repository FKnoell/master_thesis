from agents.actor_agent import ActorAgent


class PowerDecimaAgent(ActorAgent):
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims,
                 output_dim, max_depth, executor_levels, eps=1e-6,
                 act_fn=None, optimizer=None, scope='actor_agent',
                 pidle=0.0, pdyn=1.0, carbon_schedule=None):
        kwargs = {
            'optimizer': optimizer,
            'scope': scope,
            'pidle': pidle,
            'pdyn': pdyn,
            'carbon_schedule': carbon_schedule,
        }
        if act_fn is not None:
            kwargs['act_fn'] = act_fn

        super().__init__(
            sess, node_input_dim + 3, job_input_dim + 3, hid_dims,
            output_dim, max_depth, executor_levels, eps=eps, **kwargs)
