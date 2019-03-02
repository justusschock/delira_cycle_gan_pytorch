from cycle_gan import AdversarialLoss, CycleLoss, DiscriminatorLoss

import torch


def test_adversarial_loss():
    loss_fn = AdversarialLoss()

    # test, if loss computation works
    loss_value = loss_fn(torch.rand(20, 1), torch.rand(20, 1))

    # test if scalar value (.item() works on scalar tensors only)
    tmp = loss_value.item()


def test_cycle_loss():
    loss_fn = CycleLoss()

    # test, if loss computation works
    loss_value = loss_fn(torch.rand(20, 1, 224, 224),
                         torch.rand(20, 1, 220, 220),
                         torch.rand(20, 1, 224, 224),
                         torch.rand(20, 1, 220, 220))

    # test if scalar value (.item() works on scalar tensors only)
    tmp = loss_value.item()


def test_discriminator_loss():
    loss_fn = DiscriminatorLoss()

    # test, if loss computation works
    loss_value = loss_fn(torch.rand(20, 1),
                         torch.rand(20, 1))

    # test if scalar value (.item() works on scalar tensors only)
    tmp = loss_value.item()
