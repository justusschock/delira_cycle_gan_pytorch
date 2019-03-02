import itertools
from .model import CycleGAN


def create_optimizers_cycle_gan(model: CycleGAN, optim_cls: dict,
                                **optim_params):
    """
    Creates optimizers (one for both generators and one per discriminator) 
    holding the models' parameters

    Parameters
    ----------
    model : CycleGAN
        the model, whose parameters should be optimized
    optim_cls : dict
        dictionary containing the classes to create optimizers for the 
        generator and the discriminator
    **optim_params :
        additional parameters to create the optimizers

    Returns
    -------
    dict
        dictionary containing the different optimizers
    """

    return {
        "gen": optim_cls["gen"](itertools.chain(model.gen_a.parameters(),
                                                model.gen_b.parameters()),
                                **optim_params["gen"]),
        "discr_a": optim_cls["discr"](model.discr_a.parameters(),
                                      **optim_params["discr"]),
        "discr_b": optim_cls["discr"](model.discr_b.parameters(),
                                      **optim_params["discr"])
    }
