import numpy as np
from delira.data_loading import AbstractDataset
from delira.data_loading.load_utils import default_load_fn_2d
import os


class ImagePool:
    """
    Class to buffer images on CPU to avoid huge memory overload enable the model
    to revisit already seen images. This should prevent them to be forgotten
    """

    def __init__(self, poolsize=50, p=0.5):
        """

        Parameters
        ----------
        poolsize : int, optional
            the size of the image pool (the default is 50),
        p : float, optional
            the probability with which an image of the pool should be returned

        """

        self.pool_size = poolsize
        self.prob = p

        if self.pool_size:
            self.num_imgs = 0
            self.images = []

    def __call__(self, image):
        """
        With a given probability pushes and image to the pool and pops another
        randomly selected image
        With (1 - given probability) returns the given image

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            the image to be pushed to the pool

        Returns
        -------
        :class:`numpy.ndarray`
            the returned image (either a randomly selected one of the pool or
            the given image)

        """

        if not self.pool_size:
            return image

        if self.num_imgs < self.pool_size:
            self.num_imgs = self.num_imgs + 1
            self.images.append(image)
            return image
        else:
            p = np.random.uniform(0, 1, 1)
            if p < self.prob:
                # replace random image in
                # self.images with currently considered image
                # and return replaced image
                random_id = np.random.randint(0, self.pool_size-1)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image
                return tmp
            else:
                # return currently considered image
                return image


class UnPairedDataset(AbstractDataset):
    """
    Dataset to whold 2 unpaired datasets and returning samples from both of 
    them.

    Contains an Image pool for each dataset.

    See Also
    --------
    :class:`UnPairedRandomSampler`
    :class:`ImagePool`

    """

    def __init__(self, img_path_a, img_path_b, pool_size, pool_prob, img_size,
                 n_channels, img_extensions=[".png", ".jpg"],
                 load_fn=default_load_fn_2d):
        """

        Parameters
        ----------
        img_path_a : str
            the directory path containing the data for image domain A
        img_path_b : str
            the directory path containing the data for image domain B
        pool_size : int
            the size for both image pools
        pool_prob : float
            the sampling probability for both image pools
        img_size : int
            the size of the input image
        n_channels : int
            the number of image channels
        img_extensions : list, optional
            valid file extensions for images (the default is [".png", ".jpg"])
        load_fn : optional
            function to load a single sample (the default is 
            ``default_load_fn_2d``)

        See Also
        --------
        :class:`delira.data_loading.dataset.AbstractDataset`

        """

        super().__init__((img_path_a, img_path_b), load_fn, img_extensions,
                         None)

        self.img_shape = (img_size, img_size)
        self.n_channels = n_channels
        self.pool_a = ImagePool(pool_size, pool_prob)
        self.pool_b = ImagePool(pool_size, pool_prob)

        def _get_files(path, extensions):
            data = []
            for file in os.listdir(path):
                abs_file = os.path.join(path, file)
                if (any([abs_file.endswith(_ext) for _ext in extensions])
                        and os.path.isfile(abs_file)):
                    data.append(abs_file)
            return data

        self.data.append(_get_files(self.data_path[0], self._img_extensions))
        self.data.append(_get_files(self.data_path[1], self._img_extensions))

    def get_sample_from_index(self, index):
        """
        Returns the actual data given from an index
        Since this dataset contains two unpaired datasets, the index should be
        a tuple of ints containing the actual indices

        If the index is out of bounds for the corresponding datasets it 
        restarts at 0

        Parameters
        ----------
        index : tuple
            a tuple of ints containing the actual indices

        Returns
        -------
        tuple
            the samples from both dataset as specified by ``index``

        """

        return tuple(self.data[i][_index % len(self.data[i])]
                     for i, _index in enumerate(index))

    def __getitem__(self, index):
        """
        Returns a sample from a given index.
        Since we have the dataset consists of 2 unpaired datasets, the index
        should actually be a tuple of 2 indices

        Parameters
        ----------
        index : tuple
            a tuple of ints containing the actual indices

        Returns
        -------
        dict
            A dict with keys "domain_a" and "domain_b" containing unpaired 
            images as :class:`numpy.ndarray`

        """

        samples = self.get_sample_from_index(index)

        domain_a = self._load_fn(samples[0], img_shape=self.img_shape,
                                 n_channels=self.n_channels)["data"]

        domain_b = self._load_fn(samples[1], img_shape=self.img_shape,
                                 n_channels=self.n_channels)["data"]

        return {"domain_a": self.pool_a(domain_a),
                "domain_b": self.pool_b(domain_b)}

    def __len__(self):
        """
        Returns the length of the whole dataset as the maximum of the 
        subdatasets' lengths

        Returns
        -------
        int
            the dataset length

        """

        return max([len(_data) for _data in self.data])
