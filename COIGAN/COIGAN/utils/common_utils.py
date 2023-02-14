import numbers
import torch

import logging
import warnings

LOGGER = logging.getLogger(__name__)


def requires_grad(module, value=True):
    """
    Set requires_grad attribute of all parameters in module to value
    """
    for param in module.parameters():
        param.requires_grad = value


def accumulate(model1, model2, decay=0.999):
    """
    Used for the EMA model (Exponential Moving Average)
    in certain cases the ema model have a better performance than the original model.
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def make_optimizer(model, kind='adamw', **kwargs):

    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(model.parameters(), **kwargs)


def sample_data(loader):
        while True:
            for batch in loader:
                yield batch

def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")


def sum_dict_with_prefix(target, cur_dict, prefix, default=0):
    for k, v in cur_dict.items():
        target_key = prefix + k
        target[target_key] = target.get(target_key, default) + v


def average_dicts(dict_list):
    result = {}
    norm = 1e-3
    for dct in dict_list:
        sum_dict_with_prefix(result, dct, '')
        norm += 1
    for k in list(result):
        result[k] /= norm
    return result


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


def flatten_dict(dct):
    result = {}
    for k, v in dct.items():
        if isinstance(k, tuple):
            k = '_'.join(k)
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_dict(v).items():
                result[f'{k}_{sub_k}'] = sub_v
        else:
            result[k] = v
    return result


def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))

