import torch
from collections import OrderedDict

def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def save_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'use_rnn': net.use_rnn,
        },
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)

def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net