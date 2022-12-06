import numbers
import torch

import logging
import warnings

LOGGER = logging.getLogger(__name__)


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


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


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

