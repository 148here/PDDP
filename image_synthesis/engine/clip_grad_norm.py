from torch.nn.utils import clip_grad_norm_


class ClipGradNorm(object):
    def __init__(self, 
                 start_iteration=0, 
                 end_iteration=-1, # if negative, the norm will be always clipped
                 max_norm=0.5):
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.max_norm = max_norm
    
        self.last_epoch = -1


    def __call__(self, parameters, step=None):
        self.last_epoch = self.last_epoch + 1 if step is None else int(step)
        clip = self.last_epoch >= self.start_iteration and (
            self.end_iteration < 0 or self.last_epoch < self.end_iteration
        )
        if clip:
            clip_grad_norm_(parameters, max_norm=self.max_norm)
        return clip

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}
    

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
