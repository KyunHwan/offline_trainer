import torch
import torch.nn as nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Noise_Latent_Critic_Trainer(nn.Module):
    def __init__(self, 
                 action_critic, 
                 noise_latent_critic,
                 flow_matching_policy,
                 lr: float=3e-4):
        # takes in noise critic & action critic 
        super().__init__()
        self.Q = action_critic
        self.noise_latent_Q = noise_latent_critic
        self.flow_matching_policy = flow_matching_policy
        self.lr = lr
        self.optimizer = self._make_optimizer()

        self.loss_func = nn.MSELoss()

    def forward(self, qpos, images):
        """
        qpos: batch, num_obs, robot_state_dim
        """
        if len(qpos.shape) == 2:
            batch, _ = qpos.shape
        else:
            batch, _, _ = qpos.shape
        actions_hat = torch.randn(batch, 40, 24).cuda()
        Q_output = None
        with torch.no_grad():
            policy_actions = self.flow_matching_policy(
                                robot_state=qpos, 
                                image=images, 
                                actions_hat = actions_hat
                              )
            Q_output = self.Q(qpos, 
                              images, 
                              policy_actions
                              )
            
        loss = self.loss_func(self.noise_latent_Q(qpos, images, actions_hat), Q_output)
        return loss

    def _make_optimizer(self):
        """
        Build and return an optimizer that ONLY updates noise_latent_Q.
        opt_type: "radam_free" or "adamw_free"
        """
        params = self.noise_latent_Q.parameters()
        return radam_free(
            params,
            lr=self.lr,
            betas=(0.95, 0.999),
            silent_sgd_phase=True
        )
    
    def serialize(self):
        return self.noise_latent_Q.state_dict()

    def deserialize(self, model_dict):
        return self.noise_latent_Q.load_state_dict(model_dict, strict=True)

    def serialize_optimizer(self):
        return self.optimizer.state_dict()
    
    def deserialize_optimizer(self, optimizer_dict):
        return self.optimizer.load_state_dict(optimizer_dict)
