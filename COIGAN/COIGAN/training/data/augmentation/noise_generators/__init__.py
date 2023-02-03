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