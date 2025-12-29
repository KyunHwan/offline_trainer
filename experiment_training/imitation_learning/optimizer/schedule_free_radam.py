import torch
from torch import nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from offline_trainer.registry import OPTIMIZER_BUILDER_REGISTRY

@OPTIMIZER_BUILDER_REGISTRY.register("schedule_free_radam")
class RAdamScheduleFreeFactory(nn.Module):
    def build(self, 
              params,
              lr,
              betas=(0.9, 0.999),
              silent_sgd_phase=True
             ) -> torch.optim.Optimizer:
        
        return radam_free(params, 
                          lr=lr, 
                          betas=betas, 
                          silent_sgd_phase=silent_sgd_phase)