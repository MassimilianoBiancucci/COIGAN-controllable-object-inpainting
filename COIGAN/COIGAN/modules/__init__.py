import logging

from COIGAN.modules.ffc import FFCResNetGenerator
from COIGAN.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, NLayerDiscriminator, MultidilatedNLayerDiscriminator
from COIGAN.modules.stylegan2.swagan import Generator as SwaganGenerator
from COIGAN.modules.stylegan2.swagan import Discriminator as SwaganDiscriminator

def make_generator(kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)
    
    if kind == 'swagan_discriminator':
        return SwaganDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')