from COIGAN.training.data.augmentation.noise_generators.base_noise_generator import BaseNoiseGenerator


def make_noise_generator(
    kind: str,
    kind_kwargs,
) -> BaseNoiseGenerator:
    """
    Make a noise generator

    Args:
        kind (str): kind of noise generator
        **kwargs: parameters of the noise generator

    Returns:
        NoiseGenerator: noise generator
    """
    if kind == 'gaussian':
        from COIGAN.training.data.augmentation.noise_generators.gaussian_noise_generator import GaussianNoiseGenerator
        return GaussianNoiseGenerator(**kind_kwargs)
    if kind == 'multiscale':
        from COIGAN.training.data.augmentation.noise_generators.multiscale_noise_generator import MultiscaleNoiseGenerator
        return MultiscaleNoiseGenerator(**kind_kwargs)

    raise ValueError(f'Unknown noise generator kind {kind}')


##########################################
### DEBUG

if __name__ == "__main__":
    
    import torch
    from tqdm import tqdm
    from COIGAN.utils.debug_utils import check_nan

    multiscale_conf = {
        "kind": "multiscale",
        "kind_kwargs":{
            "interpolation": "bilinear",
            "strategy": "replace",
            "scales": [1, 3, 6, 12, 24],
            "base_generator_kwargs":{
                "kind": "gaussian",
                "kind_kwargs":{
                    "mean": 0.0,
                    "std": 0.1
                }
            }
        }
    }

    noise_gen = make_noise_generator(**multiscale_conf)

    mask = torch.zeros((4, 256, 256))
    #mask[:, 128:, :] = 1

    for _ in tqdm(range(int(1e7))):
        noise = check_nan(noise_gen(mask))
