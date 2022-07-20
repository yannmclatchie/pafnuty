"""Random number generators module."""
import numpy as np


class GGL:
    def __init__(self, m=2**31 - 1, a=16_807, c=0, seed=300416):
        """Initialise GGL settings, defaulting to MLGC settings."""
        self.m = m
        self.a = a
        self.c = c
        self.seed = seed

    def _ggl(self, x):
        """GGL random number generator.

        Args:
            x (int): sample from previous RNG iteration

        Yields:
            int: RNG sample iterations

        """

        while True:
            # GGL sampling from previous iteration
            x = (self.a * x + self.c) % self.m
            yield x

    def sample(self, N=10**6):
        """Sample from GGL RNG.

        Args:
            N (int): number of samples to return

        Returns:
            ndarray: N samples from GGL RNG

        Examples:
            >>> import pafnuty.samplers as samplers
            >>> # initialise RNG object with default settings
            >>> rng = samplers.GGL()
            >>> # sample 100 data points from the RNG
            >>> rng.sample(N=100)

        """

        # initialise RNG
        lcg = self._ggl(self.seed)
        # return samples from the MLGC RNG
        return np.array([next(lcg) for i in range(N)])

    def norm_sample(self, lower=0, upper=1, N=10**6):
        """Normalised samples from GGL RNG.

        Args:
            lower (float): the lower bound of uniform samples
            upper (float): the upper bound of uniform samples
            N (int): number of samples to return

        Returns:
            ndarray: N normalised samples from GGL RNG

        Examples:
            >>> import pafnuty.samplers as samplers
            >>> # initialise RNG object with default settings
            >>> rng = samplers.GGL()
            >>> # sample 100 normalised data points from the RNG
            >>> rng.norm_sample(N=100)

        """

        # verify bounds
        if upper <= lower:
            raise ValueError(
                "Upper bound must be strictly greater than the lower bound."
            )
        # make vanilla samples
        samples = self.sample(N=N)
        # return normalised samples
        norm_samples = np.array(samples) / self.m
        return norm_samples * (upper - lower) + lower


class LFG:
    def __init__(self, seeds=None, m=2**32, a=24, b=55):
        """ "Initialise LFG RNG class with RAN3 settings."""

        self.m = m
        self.a = a
        self.b = b
        if seeds:
            # check that there are sufficiently many elements in initial seeds
            if len(seeds) < self.b:
                raise ValueError(
                    f"Initial seeds must contain at least {self.b} elements."
                )

            self.seeds = seeds
        else:
            # initialise seeds from GGL RGN
            ggl = GGL()
            self.seeds = ggl.sample(N=self.b)

    def _lfg(self):
        """LFG random number generator, with RAN3 settings.

        Yields:
            int: iteratively appends to seeds list, and returns new last element

        """

        while True:
            self.seeds = np.append(
                self.seeds, (self.seeds[-self.b] - self.seeds[-self.a]) % self.m
            )
            yield self.seeds[-1]

    def sample(self, N=10**6):
        """Samples from LFG RNG.

        Args:
            N (int): number of samples to return

        Returns:
            ndarray: N samples from LFG RNG

        Examples:
            >>> import pafnuty.samplers as samplers
            >>> # initialise RNG object with default settings
            >>> rng = samplers.LFG()
            >>> # sample 100 data points from the RNG
            >>> rng.sample(N=100)

        """

        rng = self._lfg()
        return np.array([next(rng) for i in range(N)])

    def norm_sample(self, lower=0, upper=1, N=10**6):
        """Normalised samples from LFG RNG.

        Args:
            lower (float): the lower bound of uniform samples
            upper (float): the upper bound of uniform samples
            N (int): number of samples to return

        Returns:
            ndarray: N normalised samples from LFG RNG

        Examples:
            >>> import pafnuty.samplers as samplers
            >>> # initialise RNG object with default settings
            >>> rng = samplers.LFG()
            >>> # sample 100 normalised data points from the RNG
            >>> rng.norm_sample(N=100)

        """

        # verify bounds
        if upper <= lower:
            raise ValueError(
                "Upper bound must be strictly greater than the lower bound."
            )
        # make vanilla samples
        samples = self.sample(N=N)
        # return normalised samples
        norm_samples = np.array(samples) / self.m
        return norm_samples * (upper - lower) + lower
