import torch

def find_null_grads(model):
    """Method that loking for parameters with null gradients.

    Args:
        model (nn.Module): model to inspect.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)


def there_is_nan(input):
    """Method that checks if there is a NaN in a dictionary.

    Args:
        dict (Tensor, dict[Tensor], list[Tensor]): dictionary to inspect.

    Returns:
        bool: True if there is a NaN, False otherwise.
    """
    if isinstance(input, torch.Tensor):
        return torch.isnan(input).any()

    elif isinstance(input, dict):
        return any(
            torch.isnan(value).any() 
            for _, value in input.items()
        )

    elif isinstance(input, list):
        return any(
            torch.isnan(value).any() 
            for value in input
        )

def check_nan(input):
    if there_is_nan(input):
        raise ValueError("NaN found in the input")
    return input