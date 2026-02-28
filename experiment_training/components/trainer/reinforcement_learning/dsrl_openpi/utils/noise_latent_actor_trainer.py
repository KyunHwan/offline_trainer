import torch 
import torch.nn as nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Noise_Latent_Actor_Trainer(nn.Module):
    def __init__(self, 
                 noise_latent_critic, 
                 noise_latent_actor,
                 lr: float=3e-4):
        super().__init__()
        self.noise_latent_Q = noise_latent_critic
        self.noise_latent_actor = noise_latent_actor
        self.lr = lr
        self.optimizer = self._make_optimizer()


    def forward(self, qpos, images):
        # need to prevent optimization to occur to noise_latent_critic such that backpropagation
        # doesn't happen for noise_latent_critic. 
        # and only allow backpropagation to happen for noise_latent_actor, 
        # such that only the weights for noise_latent_actor are updated
        loss = self.noise_latent_Q(qpos, images, self.noise_latent_actor(qpos, images)) * -1.0
        return loss.mean()

    def _make_optimizer(self):
        """
        Build and return an optimizer that ONLY updates noise_latent_Q.
        opt_type: "radam_free" or "adamw_free"
        """
        params = self.noise_latent_actor.parameters()
        return radam_free(
            params,
            lr=self.lr,
            betas=(0.95, 0.999),
            silent_sgd_phase=True
        )
    
    def serialize(self):
        return self.noise_latent_actor.state_dict()

    def deserialize(self, model_dict):
        return self.noise_latent_actor.load_state_dict(model_dict, strict=True)

    def serialize_optimizer(self):
        return self.optimizer.state_dict()
    
    def deserialize_optimizer(self, optimizer_dict):
        return self.optimizer.load_state_dict(optimizer_dict)
        
