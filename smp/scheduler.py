from importlib import import_module


def create_scheduler(scheduler_name, **kwargs):
    if scheduler_name is None:
        return None
    try:
        create_fn = getattr(import_module("torch.optim.lr_scheduler"), scheduler_name)
    except AttributeError:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    scheduler = create_fn(**kwargs)
    return scheduler
