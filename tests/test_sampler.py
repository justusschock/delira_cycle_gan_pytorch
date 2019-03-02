from cycle_gan import UnPairedRandomSampler


class SimpleDset(object):
    def __init__(self):
        self.data_a = [1] * 500
        self.data_b = [0] * 1000

    def __getitem__(self, index):
        return self.data_a[index[0]], self.data_b[index[1]]

    def __len__(self):
        return max(len(self.data_a), len(self.data_b))


def test_sampler():
    sampler = UnPairedRandomSampler(list(range(20)))
    assert len(sampler) == 20
    indices = sampler(5)
    assert len(indices) == 5
    assert all([isinstance(_index, tuple) for _index in indices])

    dset = SimpleDset()

    sampler = UnPairedRandomSampler.from_dataset(dset)
    assert len(sampler) == len(dset)

    indices = sampler(5)
    assert len(indices) == 5
    assert all([isinstance(_index, tuple) for _index in indices])

    assert any([_index[0] != _index[1] for _index in indices])
