import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Action_Critic_Trainer(nn.Module):
    def __init__(self, 
                 action_critic,
                 flow_matching_policy,
                 noise_latent_actor,
                 target_update_rate: float=0.005,
                 discount_factor: float=0.99,
                 lr: float=3e-4
                 ):
        # takes in noise critic & action critic 
        super().__init__()
        self.target_update_rate = target_update_rate
        self.discount_factor = discount_factor

        self.Q = action_critic
        self.Q_target = self._make_target(action_critic).cuda()

        self.flow_matching_policy = flow_matching_policy
        self.noise_latent_actor = noise_latent_actor

        self.loss_func = nn.MSELoss()

        self.lr = lr
        self.optimizer = self._make_optimizer()

    def forward(self, qpos, images, actions, reward, next_qpos, next_images):
        Q_target_output = None
        with torch.no_grad():
            actions_hat = self.noise_latent_actor(next_qpos,next_images)
            flow_matching_actions = self.flow_matching_policy(robot_state = next_qpos, 
                                                              image = next_images,
                                                              actions_hat = actions_hat)
            Q_target_output = self.Q_target(next_qpos, 
                                            next_images, 
                                            flow_matching_actions)
            
        loss = self.loss_func(self.Q(qpos, images, actions), reward + (self.discount_factor ** 40) * Q_target_output)
        return loss
    
    def _make_target(self, net: torch.nn.Module) -> torch.nn.Module:
        tgt = copy.deepcopy(net)
        tgt.eval()                              # no dropout/bn updates in eval mode
        for p in tgt.parameters():
            p.requires_grad_(False)             # do not track grads
        return tgt

    @torch.no_grad()
    def update_target(self) -> None:
        """Polyak averaging: target <- tau*source + (1-tau)*target."""
        for p_src, p_tgt in zip(self.Q.parameters(), self.Q_target.parameters()):
            p_tgt.data.mul_(1.0 - self.target_update_rate).add_(self.target_update_rate * p_src.data)

    def _make_optimizer(self):
        """
        Build and return an optimizer that ONLY updates noise_latent_Q.
        opt_type: "radam_free" or "adamw_free"
        """
        params = self.Q.parameters()
        return radam_free(
            params,
            lr=self.lr,
            betas=(0.95, 0.999),
            silent_sgd_phase=True
        )
    
    def serialize(self):
        return self.Q.state_dict()

    def deserialize(self, model_dict):
        return self.Q.load_state_dict(model_dict, strict=True)

    def serialize_optimizer(self):
        return self.optimizer.state_dict()
    
    def deserialize_optimizer(self, optimizer_dict):
        return self.optimizer.load_state_dict(optimizer_dict)
    
        
