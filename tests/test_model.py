from cycle_gan import CycleGAN
import torch


class MiniGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, input_tensor):
        return self.model(input_tensor)


class MiniDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1, stride=2),  # to 32x32
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, padding=1, stride=2),  # to 16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3, padding=1, stride=2),  # to 8x8
            torch.nn.ReLU()
        )
        self.model_fc = torch.nn.Sequential(
            torch.nn.Linear(16 * 8 * 8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, input_tensor):
        return self.model_fc(
            self.model_conv(input_tensor).view(input_tensor.size(0), -1))


def test_model():

    model = CycleGAN(generator_cls=MiniGenerator, gen_kwargs={
                     "domain_a": {}, "domain_b": {}, "shared": {}},
                     discriminator_cls=MiniDiscriminator, discr_kwargs={
                         "domain_a": {}, "domain_b": {}, "shared": {}})

    # check single forward
    preds = model(torch.rand(10, 3, 64, 64), torch.rand(10, 3, 64, 64))
    assert len(preds) == 8
    assert all([isinstance(_pred, torch.Tensor) for _pred in preds])

    # check closure without optimizers and criterions
    model.closure(model, {"input_a": torch.rand(10, 3, 64, 64),
                          "input_b": torch.rand(10, 3, 64, 64),
                          "target_a": torch.rand(10, 3, 64, 64),
                          "target_b": torch.rand(10, 3, 64, 64)},
                  optimizers={},
                  losses={k: lambda *x: sum([_x.sum() for _x in x])
                          for k in ["cycle", "adv", "discr"]})

    # check forward with optimizers and criterions
    model.closure(
        model, {"input_a": torch.rand(10, 3, 64, 64),
                "input_b": torch.rand(10, 3, 64, 64),
                "target_a": torch.rand(10, 3, 64, 64),
                "target_b": torch.rand(10, 3, 64, 64)},
        optimizers={
            "gen": torch.optim.Adam(
                list(model.gen_a.parameters())
                + list(model.gen_b.parameters())),
            "discr_a": torch.optim.Adam(
                model.discr_a.parameters()),
            "discr_b": torch.optim.Adam(
                model.discr_b.parameters()
            )},
        losses={k: lambda *x: sum([_x.sum() for _x in x])
                for k in ["cycle", "adv", "discr"]})
