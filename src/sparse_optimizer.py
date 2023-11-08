from transformers import AdamW 
from torch.optim import Optimizer
import torch
import math
import numpy as np

class SparseAdamW(AdamW):
    def __init__(self,
                sparse_lambda = 0.1,
                lambda_schedule = None,
                max_lambda = None,
                lambda_num = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.sparse_lambda = sparse_lambda
        print(f"lambda in optimizer={self.sparse_lambda}")
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self._build_lambda_list(max_lambda, lambda_num)
    
    def _build_lambda_list(self, max_lambda, lambda_num):
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
        if self.lambda_schedule == "linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using log_linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.log(np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num))
        elif self.lambda_schedule == "exp_linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using exp_linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.exp(np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num))
        else:
            raise NotImplementedError
    
    def step_lambda(self):
        if self._lambdas is None:
            print("no lambda schedule is specified, do nothing")
            return
        else:
            if self.lambda_idx < len(self._lambdas) - 1:
                self.lambda_idx += 1
                self.sparse_lambda = self._lambdas[self.lambda_idx]
                print(f"use lambda={self.sparse_lambda}")
            else:
                print(f"reach end of self._lambdas, keep using lambda={self.sparse_lambda}")

    
    def step(self, closure = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                
                # params with sparsity regularization do not need weight decay
                # still hard to decide: which quantity stands for $\eta$ in Adam? group['lr] or stepsize?
                to_add = torch.div(exp_avg, denom) * (-step_size)
                if group["weight_decay"] > 0.0:
                    # p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
                    to_add = to_add + (-group["lr"] * group["weight_decay"]) * p.data
                p.data.add_(to_add) 


                if self.sparse_lambda > 0:
                    p.data[p.data > self.sparse_lambda] -= self.sparse_lambda
                    p.data[p.data < -self.sparse_lambda] += self.sparse_lambda
                    p.data[abs(p.data) < self.sparse_lambda] = 0.0
                
        return loss
