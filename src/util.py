import torch
from torch.nn.parallel import DistributedDataParallel
from transformers.trainer_pt_utils import get_parameter_names
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
GATE_PARAM_NAME= "lora.gate"

def compute_trainable_sparse_param(model):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    total_trainable_param = 0
    deduct = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if GATE_PARAM_NAME in n:
                deduct += (torch.numel(p) - torch.count_nonzero(p)) * model.config.hidden_size * 2  # zero_number * 768 * 2
            else:
                total_trainable_param += torch.numel(p)
    sparse_trainable_param = total_trainable_param - deduct
    return sparse_trainable_param, total_trainable_param

def create_optimizer_and_scheduler(args, model, num_training_steps: int):
    """
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
    and/or :obj:`create_scheduler`) in a subclass.
    """
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, num_training_steps=num_training_steps, optimizer=optimizer)
    return optimizer, scheduler

def create_optimizer(args, model):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    print(f"removing {GATE_PARAM_NAME} from standard optimizer")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and GATE_PARAM_NAME not in n and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and GATE_PARAM_NAME not in n and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(args, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps)
    return lr_scheduler
