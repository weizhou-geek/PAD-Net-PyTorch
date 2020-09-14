import os
import torch

def save_model(model, checkpoint, num, is_epoch=True):
    if not os.path.exists(checkpoint):
        os.system('mkdir -p '+ checkpoint)
    if is_epoch:
        torch.save(model.state_dict(), os.path.join(checkpoint, 'epoch_{:04d}.pth'.format(num)))
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint, 'iteration_{:09d}.pth'.format(num)))

def load_model(model, resume):
    pretrained_dict = torch.load(resume)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
