from cycle_gan import create_optimizers_cycle_gan
import torch


class SimpleDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_a = torch.nn.Conv2d(3, 3, 1)
        self.gen_b = torch.nn.Conv2d(3, 3, 1)
        self.discr_a = torch.nn.Conv2d(3, 3, 1)
        self.discr_b = torch.nn.Conv2d(3, 3, 1)

    def forward(self, input_tensor):
        return self.gen_a(input_tensor), self.gen_b(input_tensor), \
            self.discr_a(input_tensor), self.discr_b(input_tensor)


def test_optim_creator():
    optims = create_optimizers_cycle_gan(
        SimpleDummyModel(),
        {
            "gen": torch.optim.Adam,
            "discr": torch.optim.SGD
        },
        **{
            "discr": {"lr": 1e-3},
            "gen": {}
        }
    )

    assert isinstance(optims, dict)
    assert "gen" in optims.keys()
    assert "discr_a" in optims.keys()
    assert "discr_b" in optims.keys()
    assert isinstance(optims["gen"], torch.optim.Adam)
    assert isinstance(optims["discr_a"], torch.optim.SGD)
    assert isinstance(optims["discr_b"], torch.optim.SGD)
