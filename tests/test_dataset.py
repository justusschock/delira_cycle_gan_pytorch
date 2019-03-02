from cycle_gan import UnPairedDataset
from skimage.io import imsave
import numpy as np
import os


def test_dataset():
    data_paths = []

    path_a = os.path.join(os.path.split(__file__)[0], "test_dset", "A")
    path_b = os.path.join(os.path.split(__file__)[0], "test_dset", "B")
    os.makedirs(path_a)
    os.makedirs(path_b)

    for i in range(25):
        imsave(os.path.join(path_a, "image_%03d.png" %
                            i), np.random.rand(224, 224, 3))

        data_paths.append(os.path.join(path_a, "image_%03d.png" %
                                       i))

    for i in range(30):
        imsave(os.path.join(path_b, "image_%03d.png" %
                            i), np.random.rand(224, 224, 3))
        data_paths.append(os.path.join(path_b, "image_%03d.png" %
                                       i))

    dset = UnPairedDataset(path_a, path_b, 50, 0.5, img_size=64, n_channels=1)

    assert len(dset) == 30

    assert isinstance(dset[(5, 3)], dict)
    assert dset[(5, 3)]["domain_a"].shape == (1, 64, 64)

    try:
        dset[5]
        assert False, "should raise TypeError since index must be iterable"

    except TypeError:
        assert True

    for file in data_paths:
        os.remove(file)

    os.rmdir(path_a)
    os.rmdir(path_b)
    os.rmdir(os.path.join(os.path.split(__file__)[0], "test_dset"))
