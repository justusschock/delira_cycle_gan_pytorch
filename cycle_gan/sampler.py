from delira.data_loading.sampler import AbstractSampler, RandomSampler


class UnPairedRandomSampler(AbstractSampler):
    """
    Sampler to return two randomly sampled indices (one for each subset).
    Internally holds a single 
    :class:`delira.data_loading.sampler.random_sampler.RandomSampler` per subset

    """

    def __init__(self, indices):
        """

        Parameters
        ----------
        indices : list
            a list of integers containing all valid indices

        """

        super().__init__()
        self.sampler_a = RandomSampler(indices)
        self.sampler_b = RandomSampler(indices)

    def _get_indices(self, n_indices):
        """
        Returns the indices
        
        Parameters
        ----------
        n_indices : int
            the number of indices to return
        
        Returns
        -------
        list
            list of tuple of ints containing the indices for all subsets.
        """

        return list(zip(self.sampler_a(n_indices), self.sampler_b(n_indices)))

    def __len__(self):
        
        return max(len(self.sampler_a), len(self.sampler_b))
