""" Optimizers class """
import functools
import importlib
import math
import torch
import torch.optim as optim
import types
from collections import Counter
from math import sqrt
from mammoth.utils.misc import fn_args
from torch.nn.utils import clip_grad_norm_
from mammoth.utils.logging import logger


def attention_bridge_optimizer(model, task_queue_manager, base_optimizer):
    suboptimizers = {}
    my_grouped_components = task_queue_manager.get_grouped_components(model)
    for component_type in my_grouped_components:
        for component_id, component in my_grouped_components[component_type].items():
            if isinstance(component_id, str):
                name = component_type + '_' + component_id
            else:
                name = component_type + '_' + '_'.join([str(x) for x in component_id])
            params = []
            for param_name, param in component.named_parameters():
                if not param.requires_grad:
                    continue
                if 'adapter' in param_name and 'adapter' not in component_type:
                    # omit adapters from base component optimizers
                    continue
                if 'embedding' in param_name:
                    print(f'adding {param_name} to suboptimizer {name}')
                params.append(param)
            if name in suboptimizers:
                raise Exception(f'Trying to create second optimizer for "{name}"')
            if len(params) != 0:
                optimizer = base_optimizer(params)
                suboptimizers[name] = optimizer

    for generator_id in task_queue_manager.get_generators():
        generator = model.generator[f'generator_{generator_id}']
        params = []
        for name, param in generator.named_parameters():
            if not param.requires_grad:
                continue
            params.append(param)
        optimizer = base_optimizer(params)
        suboptimizers[f'generator_{generator_id}'] = optimizer

    attParam = []
    for name, param in model.attention_bridge.named_parameters():
        if not param.requires_grad:
            continue
        attParam.append(param)

    # skip AB optimizer if AB is not in use
    if len(attParam):
        optimizer = base_optimizer(attParam)
        suboptimizers["attention_bridge"] = optimizer

    optimizer = MultipleOptimizer(suboptimizers, None)
    return optimizer


def build_torch_optimizer(model, opts, task_queue_manager):
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well

    Args:
      model: The model to optimize.
      opts. The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = [opts.adam_beta1, opts.adam_beta2]
    if opts.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opts.learning_rate)
    elif opts.optim == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=opts.learning_rate,
            initial_accumulator_value=opts.adagrad_accumulator_init,
        )
    elif opts.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=opts.learning_rate)
    elif opts.optim == 'adafactor':
        optimizer = attention_bridge_optimizer(
            model,
            task_queue_manager,
            lambda params: AdaFactorFairSeq(params, weight_decay=opts.weight_decay),
        )
    elif opts.optim == 'adam':
        optimizer = attention_bridge_optimizer(
            model,
            task_queue_manager,
            lambda params: optim.Adam(
                params, lr=opts.learning_rate, betas=betas, eps=1e-9, weight_decay=opts.weight_decay
            )
        )
    elif opts.optim == 'adamw':
        optimizer = attention_bridge_optimizer(
            model,
            task_queue_manager,
            lambda params: optim.AdamW(
                params, lr=opts.learning_rate, betas=betas, eps=1e-9, weight_decay=opts.weight_decay
            )
        )
    elif opts.optim == 'fusedadam':
        # we use here a FusedAdam() copy of an old Apex repo
        optimizer = FusedAdam(params, lr=opts.learning_rate, betas=betas)
        if opts.model_dtype == 'fp16':
            import apex

            # In this case use the old FusedAdam with FP16_optimizer wrapper
            static_loss_scale = opts.loss_scale
            dynamic_loss_scale = opts.loss_scale == 0
            optimizer = apex.contrib.optimizers.FP16_Optimizer(
                optimizer, static_loss_scale=static_loss_scale, dynamic_loss_scale=dynamic_loss_scale
            )
    else:
        raise ValueError('Invalid optimizer type: ' + opts.optim)

    return optimizer


def make_learning_rate_decay_fn(opts):
    """Returns the learning decay function from options."""
    if opts.decay_method == 'noam':
        return functools.partial(noam_decay, warmup_steps=opts.warmup_steps, model_dim=opts.model_dim)
    elif opts.decay_method == 'noamwd':
        return functools.partial(
            noamwd_decay,
            warmup_steps=opts.warmup_steps,
            model_dim=opts.model_dim,
            rate=opts.learning_rate_decay,
            decay_steps=opts.decay_steps,
            start_step=opts.start_decay_steps,
        )
    elif opts.decay_method == 'rsqrt':
        return functools.partial(rsqrt_decay, warmup_steps=opts.warmup_steps)
    elif opts.decay_method == 'linear_warmup':
        return functools.partial(
            linear_warmup_decay,
            warmup_steps=opts.warmup_steps,
            rate=opts.learning_rate,
            train_steps=opts.train_steps,
        )
    elif opts.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opts.learning_rate_decay,
            decay_steps=opts.decay_steps,
            start_step=opts.start_decay_steps,
        )


def noam_decay(step, warmup_steps, model_dim):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return model_dim ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(step, warmup_steps, model_dim, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (
        model_dim ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        * rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    )


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


def linear_warmup_decay(step, warmup_steps, rate, train_steps):
    end_rate = 0.001 * rate
    if step <= warmup_steps:
        return (step / warmup_steps)
    else:
        return max(end_rate, (train_steps - step) / (train_steps - warmup_steps))


class MultipleOptimizer(object):
    """Implement multiple optimizers"""

    def __init__(self, op, multiOptims_Langs=None):
        self.optimizers = op
        self.multiOptims_Langs = multiOptims_Langs
        self._steps = Counter()

    @property
    def param_groups(self):
        param_groups = []
        for name in self.optimizers:
            optimizer = self.optimizers[name]
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self):
        """Reset the gradient of all sub-optimizers to zero"""
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

    def step(self, grad_scaler=None):
        """Step through all the suboptimizers"""
        for name in self.optimizers:
            if self._any_param_has_grad(self.optimizers[name], name):
                self._steps[name] += 1
                self.optimizers[name].step()

    @staticmethod
    def _any_param_has_grad(optimizer, name):
        for group in optimizer.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue
                if not hasattr(param, 'has_grad'):
                    # if there are parameters not tracked by the hook,
                    # then always perform the step
                    raise Exception(f'At least one parameter in {name} did not have the hook')
                if param.has_grad:
                    return True
        return False

    def report_steps(self):
        result = []
        for name in self.optimizers:
            count = self._steps[name]
            result.append(f'Optimizer "{name}" has been stepped {count} times')
        return result

    def state_dict(self):
        """Returns the state dictionary"""
        return {
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'multiOptims_Langs': self.multiOptims_Langs,
            'steps': self._steps,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer from the state dictionary"""

        # do not load any optimizer state if one component is missing
        do_load = True
        for k in state_dict["optimizers"].keys():
            if k not in self.optimizers.keys():
                do_load = False

        if do_load is True:
            for k in state_dict["optimizers"].keys():
                self.optimizers[k].load_state_dict(state_dict["optimizers"][k])
        else:
            logger.info("Some components do not match. Do not load optimizer from checkpoint.")

        self.multiOptims_Langs = state_dict["multiOptims_Langs"]
        self._steps = state_dict["steps"]


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.
    """

    def __init__(self, optimizer, learning_rate, learning_rate_decay_fn=None, max_grad_norm=None):
        """Initializes the controller.

        Args:
          optimizer: A ``torch.optim.Optimizer`` instance.
          learning_rate: The initial learning rate.
          learning_rate_decay_fn: An optional callable taking the current step
            as argument and return a learning rate scaling factor.
          max_grad_norm: Clip gradients to this global norm.
        """
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self._fp16 = None
        self._scaler = None

    @classmethod
    def from_opts(cls, model, opts, task_queue_manager, checkpoint=None):
        """Builds the optimizer from options.

        Args:
          cls: The ``Optimizer`` class to instantiate.
          model: The model to optimize.
          opts: The dict of user options.
          checkpoint: An optional checkpoint to load states from.

        Returns:
          An ``Optimizer`` instance.
        """
        optim_opt = opts
        optim_state_dict = None

        if opts.train_from and checkpoint is not None:
            optim = checkpoint['optim']
            ckpt_opt = checkpoint['opts']
            ckpt_state_dict = {}
            if isinstance(optim, Optimizer):  # Backward compatibility.
                ckpt_state_dict['training_step'] = optim._training_step
                ckpt_state_dict['decay_step'] = optim._decay_step
                ckpt_state_dict['optimizer'] = optim._optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            if opts.reset_optim == 'none':
                # Load everything from the checkpoint.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
            elif opts.reset_optim == 'all':
                # Build everything from scratch.
                pass
            elif opts.reset_optim == 'states':
                # Reset optimizer, keep options.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
                del optim_state_dict['optimizer']
            elif opts.reset_optim == 'keep_states':
                # Reset options, keep optimizer.
                optim_state_dict = ckpt_state_dict

        optimizer = cls(
            build_torch_optimizer(model, optim_opt, task_queue_manager),
            optim_opt.learning_rate,
            learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt),
            max_grad_norm=optim_opt.max_grad_norm,
        )

        if opts.model_dtype == "fp16":
            if opts.optim == "fusedadam":
                optimizer._fp16 = "legacy"
            else:
                optimizer._fp16 = "amp"
                from torch.cuda.amp import GradScaler

                optimizer._scaler = GradScaler()

        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)
        return optimizer

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    @property
    def amp(self):
        """True if use torch amp mix precision training."""
        return self._fp16 == "amp"

    def learning_rate(self):
        """Returns the current learning rate."""
        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        scale = self._learning_rate_decay_fn(self._decay_step)
        return scale * self._learning_rate

    def state_dict(self):
        return {
            'training_step': self._training_step,
            'decay_step': self._decay_step,
            'optimizer': self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored.
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad()

    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the
        backward pass."""
        if self.amp:
            self._scaler.scale(loss).backward()
        elif self._fp16 == "legacy":
            kwargs = {}
            if "update_master_grads" in fn_args(self._optimizer.backward):
                kwargs["update_master_grads"] = True
            self._optimizer.backward(loss, **kwargs)
        else:
            loss.backward()

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        learning_rate = self.learning_rate()

        if self.amp:
            for suboptimizer in self._optimizer.optimizers.values():
                self._scaler.unscale_(suboptimizer)
        elif self._fp16 == "legacy":
            if hasattr(self._optimizer, "update_master_grads"):
                self._optimizer.update_master_grads()
            if hasattr(self._optimizer, "clip_master_grads") and self._max_grad_norm > 0:
                self._optimizer.clip_master_grads(self._max_grad_norm)

        for group in self._optimizer.param_groups:
            group['lr'] = learning_rate
            if self._max_grad_norm > 0 and self._fp16 != "legacy":
                clip_grad_norm_(group['params'], self._max_grad_norm)

        if self.amp:
            self._scaler.step(self._optimizer)

            # Updates the scale for next iteration.
            self._scaler.update()
        else:
            self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1


# Code below is an implementation of https://arxiv.org/pdf/1804.04235.pdf
# inspired but modified from https://github.com/DeadAt0m/adafactor-pytorch

"""
class AdaFactor(torch.optim.Optimizer):

    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30,
                 eps2=1e-3, cliping_threshold=1, non_constant_decay=True,
                 enable_factorization=True, ams_grad=True, weight_decay=0):

        enable_momentum = beta1 != 0

        if non_constant_decay:
            ams_grad = False

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps1=eps1,
                        eps2=eps2, cliping_threshold=cliping_threshold,
                        weight_decay=weight_decay, ams_grad=ams_grad,
                        enable_factorization=enable_factorization,
                        enable_momentum=enable_momentum,
                        non_constant_decay=non_constant_decay)

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1]*shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (shape[0]*functools.reduce(operator.mul,
                                                   temp_shape[tmp_div:], 1),
                         shape[1]*functools.reduce(operator.mul,
                                                   temp_shape[:tmp_div], 1))
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        '''
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        '''
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse \
                                       gradients, use SparseAdam instead')

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group['enable_factorization']:
                    new_shape, old_shape = \
                        self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if group['enable_momentum']:
                        state['exp_avg'] = torch.zeros(new_shape,
                                                       dtype=torch.float32,
                                                       device=p.grad.device)

                    if is_matrix and group['enable_factorization']:
                        state['exp_avg_sq_R'] = \
                            torch.zeros((1, new_shape[1]),
                                        dtype=torch.float32,
                                        device=p.grad.device)
                        state['exp_avg_sq_C'] = \
                            torch.zeros((new_shape[0], 1),
                                        dtype=torch.float32,
                                        device=p.grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros(new_shape,
                                                          dtype=torch.float32,
                                                          device=p.grad.device)
                    if group['ams_grad']:
                        state['exp_avg_sq_hat'] = \
                            torch.zeros(new_shape, dtype=torch.float32,
                                        device=p.grad.device)

                if group['enable_momentum']:
                    exp_avg = state['exp_avg']

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_r = state['exp_avg_sq_R']
                    exp_avg_sq_c = state['exp_avg_sq_C']
                else:
                    exp_avg_sq = state['exp_avg_sq']

                if group['ams_grad']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']

                state['step'] += 1
                lr_t = group['lr']
                lr_t *= max(group['eps2'], self._rms(p.data))

                if group['enable_momentum']:
                    if group['non_constant_decay']:
                        beta1_t = group['beta1'] * \
                                  (1 - group['beta1'] ** (state['step'] - 1)) \
                                  / (1 - group['beta1'] ** state['step'])
                    else:
                        beta1_t = group['beta1']
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                if group['non_constant_decay']:
                    beta2_t = group['beta2'] * \
                              (1 - group['beta2'] ** (state['step'] - 1)) / \
                              (1 - group['beta2'] ** state['step'])
                else:
                    beta2_t = group['beta2']

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_r.mul_(beta2_t). \
                        add_(1 - beta2_t, torch.sum(torch.mul(grad, grad).
                                                    add_(group['eps1']),
                                                    dim=0, keepdim=True))
                    exp_avg_sq_c.mul_(beta2_t). \
                        add_(1 - beta2_t, torch.sum(torch.mul(grad, grad).
                                                    add_(group['eps1']),
                                                    dim=1, keepdim=True))
                    v = torch.mul(exp_avg_sq_c,
                                  exp_avg_sq_r).div_(torch.sum(exp_avg_sq_r))
                else:
                    exp_avg_sq.mul_(beta2_t). \
                        addcmul_(1 - beta2_t, grad, grad). \
                        add_((1 - beta2_t)*group['eps1'])
                    v = exp_avg_sq

                g = grad
                if group['enable_momentum']:
                    g = torch.div(exp_avg, 1 - beta1_t ** state['step'])

                if group['ams_grad']:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(g, (torch.div(v, 1 - beta2_t **
                                  state['step'])).sqrt().add_(group['eps1']))
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group['cliping_threshold']))
                p.data.add_(-lr_t * (u.view(old_shape) if is_need_reshape and
                            group['enable_factorization'] else u))

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * lr_t, p.data)

        return loss
"""


class FusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.
       Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        amsgrad=False,
    ):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError('AMSGrad variant not supported.')
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(self, closure=None, grads=None, output_params=None, scale=1.0, grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients.
                (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif not isinstance(grads[0], list):
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif not isinstance(output_params[0], list):
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        for group, grads_this_group, output_params_this_group, grad_norm in zip(
            self.param_groups, grads_group, output_params_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group['params'])

            # compute combined scale factor for this group
            combined_scale = scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param in zip(group['params'], grads_this_group, output_params_this_group):
                # note: p.grad should not ever be set for correct operation of
                # mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('sparse gradient not supported')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                fused_adam_cuda.adam(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    combined_scale,
                    state['step'],
                    self.eps_mode,
                    bias_correction,
                    group['weight_decay'],
                )
        return loss


class AdaFactorFairSeq(torch.optim.Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(AdaFactorFairSeq, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
