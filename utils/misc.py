def adjust_learning_rate(optimizer, init_lr, every):
    #import pdb; pdb.set_trace()
    lrd = init_lr / every
    old_lr = optimizer.param_groups[0]['lr']
    # linearly decaying lr
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


