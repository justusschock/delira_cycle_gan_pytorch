import torch


class CycleLoss(torch.nn.Module):
    """
    Computes a cyclic loss between the original and reconstructed  images of 
    each domain

    """

    def __init__(self, loss_fn=torch.nn.L1Loss()):
        """

        Parameters
        ----------
        loss_fn : optional
            the actual loss function to compute a pixelwise loss
            (the default is :class:`torch.nn.L1Loss`())

        """

        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, target_a: torch.Tensor, target_b: torch.Tensor,
                rec_a: torch.Tensor, rec_b: torch.Tensor):
        """
        Calculates the actual loss

        Parameters
        ----------
        target_a : :class:`torch.Tensor`
            the target image of domain A
        target_b : :class:`torch.Tensor`
            the target image of domain B
        rec_a : :class:`torch.Tensor`
            the reconstructed image of domain A
        rec_b : :class:`torch.Tensor`
            the reconstructed image of domain B

        Returns
        -------
        :class:`torch.Tensor`
            the loss value

        """

        return self._loss_fn(rec_a, target_a) + self._loss_fn(rec_b, target_b)


class AdversarialLoss(torch.nn.Module):
    """
    Calculates an adversarial loss on the classification results of the fake 
    images of both image domains (needed to update the generators)

    """

    def __init__(self, loss_fn=torch.nn.BCELoss()):
        """

        Parameters
        ----------
        loss_fn : optional
            the actual loss function computing the losses 
            (the default is :class:`torch.nn.BCELoss`())

        """

        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, fake_a_cls: torch.Tensor, fake_b_cls: torch.Tensor):
        """
        Calculates the actual loss

        Parameters
        ----------
        fake_a_cls : :class:`torch.Tensor`
            classification result of the fake image in domain A
        fake_b_cls : :class:`torch.Tensor`
            classification result of the fake image in domain B

        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """

        return self._loss_fn(fake_a_cls, torch.ones_like(fake_a_cls)) \
            + self._loss_fn(fake_b_cls, torch.ones_like(fake_b_cls))


class DiscriminatorLoss(torch.nn.Module):
    """
    Calculates a classical discriminator loss 
    (classification whether image is real or fake)

    """

    def __init__(self, loss_fn=torch.nn.BCELoss()):
        """

        Parameters
        ----------
        loss_fn : optional
            the actual loss function computing the losses 
            (the default is :class:`torch.nn.BCELoss`())

        """
        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, real_cl: torch.Tensor, fake_cl: torch.Tensor):
        """
        Calculates the actual loss

        Parameters
        ----------
        real_cl : :class:`torch.Tensor`
            classification result of the real image
        fake_cl : :class:`torch.Tensor`
            classification result of the fake image

        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        return self._loss_fn(real_cl, torch.ones_like(real_cl)) \
            + self._loss_fn(fake_cl, torch.zeros_like(fake_cl))
