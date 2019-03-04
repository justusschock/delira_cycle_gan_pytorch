import torch
from delira.models import AbstractPyTorchNetwork
import typing
import logging


class CycleGAN(AbstractPyTorchNetwork):
    """
    A delira-compatible implementation of a cycle-GAN.

    The Cycle GAN is an unpaired image-to-image translation.

    The image domains in this implementation are called A and B.
    The suffix of each tensor indicates the corresponding image domain (e.g.
    ``x_a`` lies within domain A, while ``x_b`` lies within domain B) and the 
    suffix of each module indicates the domain it works on (e.g. ``net_a`` 
    works on images of domain A).

    Performs predictions and has ``closure`` method to provide the basic
    behavior during training.

    Performs the following operations during prediction:

        .. math::

            fake_B = G_A(I_A)

            fake_A = G_B(I_B)

            rec_A = G_B(G_A(I_A))

            rec_B = G_A(G_B(I_B))

    and the classification results are additionally send through the
    discriminator:

        .. math::

            fake_{A,CL} = D_A(G_B(I_B))

            fake_{B,CL} = D_B(G_A(I_A))

            real_{A,CL} = D_A(I_A)

            real_{B,CL} = D_B(I_B)

    During Training a cyclic loss is computed between rec_A and I_A and
    rec_b and I_B since the reconstructed images must equal the original ones.

    Additionally a classical adversarial loss and a loss to update the
    discriminators are calculated as usual in GANs

    Note
    ----
    * The provided ``generator_cls`` should produce outputs of the same size as
        it's inputs.
    * The provided ``discriminator_cls`` should accept inputs of the same size
        as the ``generator_cls`` produces and    map them down to a single
        scalar output.

    See Also
    --------
    :class:`CycleLoss`
    :class:`AdversarialLoss`
    :class:`DiscriminatorLoss`

    References
    ----------
    https://arxiv.org/abs/1703.10593

    """

    def __init__(self, generator_cls, gen_kwargs: dict,
                 discriminator_cls, discr_kwargs: dict,
                 cycle_weight=1, adversarial_weight=1, gen_update_freq=1,
                 img_logging_freq=1):
        """

        Parameters
        ----------
        generator_cls :
            the class of the generator networks
        gen_kwargs : dict
            keyword arguments to instantiate both generators, must contain a
            subdict "shared" which should contain all configurations, which
            apply for both domains and subdicts "domain_a" and "domain_b"
            containing the domain_specific configurations
        discriminator_cls :
            the class of the discriminator networks
        discr_kwargs : dict
            keyword arguments to instantiate both discriminators, must contain a
            subdict "shared" which should contain all configurations, which
            apply for both domains and subdicts "domain_a" and "domain_b"
            containing the domain_specific configurations
        cycle_weight : int, optional
            the weight of the cyclic loss (the default is 1)
        adversarial_weight : int, optional
            the weight of the adversarial loss (the default is 1)
        gen_update_freq : int, optional
            defines, how often the generator will be updated: a frequency of 2
            means an update every 2 iterations, a frequency of 3 means an update
            every 3 iterations etc. (the default is 1, which means an update at
            every iteration)
        img_logging_freq : int, optional
            defines, how often the images will be logged: a frequency of 2
            means a log every 2 iterations, a frequency of 3 means a log
            every 3 iterations etc. (the default is 1, which means a log at
            every iteration)

        """

        super().__init__()

        self.gen_a = generator_cls(
            **gen_kwargs["domain_a"], **gen_kwargs["shared"]
        )
        self.gen_b = generator_cls(
            **gen_kwargs["domain_b"], **gen_kwargs["shared"]
        )

        self.discr_a = discriminator_cls(
            **discr_kwargs["domain_a"], **discr_kwargs["shared"]
        )

        self.discr_b = discriminator_cls(
            **discr_kwargs["domain_b"], **discr_kwargs["shared"]
        )

        self.cycle_weight = cycle_weight
        self.adversarial_weight = adversarial_weight
        self.gen_update_freq = gen_update_freq
        self.img_logging_freq = img_logging_freq

    def forward(self, input_domain_a: torch.Tensor,
                input_domain_b: torch.Tensor):
        """
        Performs all relevant predictions:

            .. math::

                fake_B = G_A(I_A)

                fake_A = G_B(I_B)

                rec_A = G_B(G_A(I_A))

                rec_B = G_A(G_B(I_B))

                fake_{A,CL} = D_A(G_B(I_B))

                fake_{B,CL} = D_B(G_A(I_A))

                real_{A,CL} = D_A(I_A)

                real_{B,CL} = D_B(I_B)

        Parameters
        ----------
        input_domain_a : :class:`torch.Tensor`
            the image batch of domain A
        input_domain_b : :class:`torch.Tensor`
            the image batch of domain B

        Returns
        -------
        :class:`torch.Tensor`
            the reconstructed images of domain A: G_B(G_A(I_A))
        :class:`torch.Tensor`
            the reconstructed images of domain B: G_A(G_B(I_B))
        :class:`torch.Tensor`
            the generated fake image in domain A: G_B(I_B)
        :class:`torch.Tensor`
            the generated fake image in domain B: G_A(I_A)
        :class:`torch.Tensor`
            the classification result of the real image of domain A: D_A(I_A)
        :class:`torch.Tensor`
            the classification result of the generated fake image in domain A:
            D_A(G_B(I_B))
        :class:`torch.Tensor`
            the classification result of the real image of domain B: D_B(I_B)
        :class:`torch.Tensor`
            the classification result of the generated fake image in domain B:
            D_B(G_A(I_A))

        """

        fake_b = self.gen_a(input_domain_a)
        fake_a = self.gen_b(input_domain_b)

        fake_a_cl = self.discr_a(fake_a)
        fake_b_cl = self.discr_b(fake_b)

        real_a_cl = self.discr_a(input_domain_a)
        real_b_cl = self.discr_b(input_domain_b)

        rec_a = self.gen_b(fake_b)
        rec_b = self.gen_a(fake_a)

        return rec_a, rec_b, fake_a, fake_b, real_a_cl, fake_a_cl, real_b_cl, \
            fake_b_cl

    @staticmethod
    def prepare_batch(batch_dict: dict,
                      input_device: typing.Union[torch.device, str],
                      output_device: typing.Union[torch.device, str]):
        """
        Pushes the necessary batch inputs to the correct device

        Parameters
        ----------
        batch_dict : dict
            the dict containing all batch elements
        input_device : :class:`torch.device` or str
            the device for al network inputs
        output_device : :class:`torch.device` or str
            the device for al network outputs

        Returns
        -------
        dict
            dictionary with all elements on correct devices and with correct
            dtype; contains the following keys:
            ['input_a', 'input_b', 'target_a', 'target_b']

        """

        return {
            "input_a": torch.from_numpy(
                batch_dict["domain_a"]).to(input_device, torch.float),
            "input_b": torch.from_numpy(
                    batch_dict["domain_b"]).to(input_device, torch.float),
            "target_a": torch.from_numpy(
                batch_dict["domain_a"]).to(output_device, torch.float),
            "target_b": torch.from_numpy(
                batch_dict["domain_b"]).to(output_device, torch.float)
        }

    @staticmethod
    def closure(model, data_dict: dict,
                optimizers: dict, losses={}, metrics={},
                fold=0, batch_nr=0, **kwargs):
        """
        closure method to do a single backpropagation step
        Parameters
        ----------
        model : :class:`CycleGAN`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the losses to calculate errors
            (gradients from different losses will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        batch_nr : int
            Number of batch in current epoch (starts with 0 at begin of every
            epoch; default: 0)
        **kwargs:
            additional keyword arguments
        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        list
            Arbitrary number of predictions as torch.Tensor
        Raises
        ------
        AssertionError
            if optimizers or losses are empty or the optimizers are not
            specified
        """

        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        lambdas = {}

        for key in ["cycle", "adversarial", "gen_freq", "img_logging_freq"]:
            if isinstance(model, torch.nn.DataParallel):
                lambdas[key] = getattr(model.module, "lambda_" + key)
            else:
                lambdas[key] = getattr(model, "lambda_" + key)

        loss_vals = {}
        metric_vals = {}

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():
            input_a, input_b = data_dict.pop(
                "input_a"), data_dict.pop("input_b")

            target_a, target_b = data_dict.pop(
                "target_a"), data_dict.pop("target_b")

            rec_a, rec_b, fake_a, fake_b, real_a_cl, fake_a_cl, real_b_cl, \
                fake_b_cl = model(input_a, input_b)

            # calculate losses

            # calculate cycle_loss
            cycle_loss = losses["cycle"](
                target_a, target_b, rec_a, rec_b) * lambdas["cycle"]

            # calculate adversarial loss
            adv_loss = losses["adv"](
                fake_a_cl, fake_b_cl)*lambdas["adversarial"]

            gen_loss = cycle_loss + adv_loss

            # calculate discriminator losses
            discr_a_loss = losses["discr"](real_a_cl, fake_a_cl)
            discr_b_loss = losses["discr"](real_b_cl, fake_b_cl)

            # assign detached losses to return dict
            loss_vals["discr_a"] = discr_a_loss.item()
            loss_vals["discr_b"] = discr_b_loss.item()
            loss_vals["adv"] = adv_loss.item()
            loss_vals["cycle"] = cycle_loss.item()
            loss_vals["gen_total"] = gen_loss.item()

            if optimizers:

                # optimize optimizer every lambdas["gen_freq"] iterations
                if (batch_nr % lambdas["gen_freq"]) == 0:
                    with optimizers["gen"].scale_loss(gen_loss) as scaled_loss:
                        optimizers["gen"].zero_grad()
                        scaled_loss.backward(retain_graph=True)
                        optimizers["gen"].step()

                # optimize discriminator a
                with optimizers["discr_a"].scale_loss(discr_a_loss) as scaled_loss:
                    optimizers["discr_a"].zero_grad()
                    scaled_loss.backward()
                    optimizers["discr_a"].step()

                # optimize discriminator b
                with optimizers["discr_b"].scale_loss(discr_b_loss) as scaled_loss:
                    optimizers["discr_b"].zero_grad()
                    scaled_loss.backward()
                    optimizers["discr_b"].step()

            else:
                # eval mode if no optimizers are given -> add prefix "val_"
                eval_loss_vals, eval_metric_vals = {}, {}

                for key, val in loss_vals.items():
                    eval_loss_vals["val_" + key] = val
                for key, val in metric_vals.items():
                    eval_metric_vals["val_" + key] = val

                loss_vals = eval_loss_vals
                metric_vals = eval_metric_vals

        if (batch_nr % lambdas["img_logging_freq"]) == 0:
            logging.info({'image_grid': {
                "images": input_a,
                "name": "input images domain A",
                "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {
                "images": input_b,
                "name": "input images domain B",
                "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {
                "images": fake_a,
                "name": "fake images domain A",
                "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {
                "images": fake_b,
                "name": "fake images domain B",
                "env_appendix": "_%02d" % fold}})

        return metric_vals, loss_vals, [rec_a, rec_b, fake_a, fake_b,
                                        real_a_cl, fake_a_cl, real_b_cl,
                                        fake_b_cl]

    @property
    def lambda_cycle(self):
        return self.cycle_weight

    @lambda_cycle.setter
    def lambda_cycle(self, new_val):
        self.cycle_weight = new_val

    @property
    def lambda_adversarial(self):
        return self.adversarial_weight

    @lambda_adversarial.setter
    def lambda_adversarial(self, new_val):
        self.adversarial_weight = new_val

    @property
    def lambda_gen_freq(self):
        return self.gen_update_freq

    @lambda_gen_freq.setter
    def lambda_gen_freq(self, new_val):
        self.gen_update_freq = new_val

    @property
    def lambda_img_logging_freq(self):
        return self.img_logging_freq

    @lambda_img_logging_freq.setter
    def lambda_img_logging_freq(self, new_val):
        self.img_logging_freq = new_val
