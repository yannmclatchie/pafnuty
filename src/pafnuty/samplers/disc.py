"""Disc sampling module."""

import numpy as np

from pafnuty.samplers.rng import LFG


class Disc:
    def __init__(self, r=1, strategy="inverse"):
        """Initialise disc sampling method.

        This method provides disc sampling capabilities with three possible
        sampling strategies, and making use of the pseudorandom number generator
        classes also defined in this package.

        Args:
            r (float): the radius of the disc to be sampled
            strategy (string): the sampling strategy to use

        Examples:
            >>> import pafnuty.samplers as samplers
            >>> import matplotlib.pyplot as plt
            >>> # initialise disc sampling object with unit radius and inverse
            >>> # transformation strategy
            >>> disc = samplers.Disc()
            >>> # sample 100 data points on disc
            >>> x, y = disc.sample(N=100)
            >>> # plot samples
            >>> plt.figure(figsize=(10, 10))
            >>> plt.plot(x, y, "o")
            >>> plt.show()

        """

        self.r = r
        self.strategy = strategy
        self.rng = LFG()

        if self.strategy not in ["rejection", "polar", "inverse"]:
            raise ValueError(
                "Invalid sampling strategy, please select one of either "
                + "`rejection`, `polar`, or `inverse`."
            )

    def _inverse(self, N=10e6):
        """Perform disc sampling using the inverse CDF strategy.

        Inverse distribution sampling will produce uniformly distributed samples
        across the disc. This is because we have directly approximated the CDF,
        and thus have avoided multiplying i.i.d. random variables. By
        identifying an invertible CDF, we are able to achieve our function, F(),
        that does not require the rejection of points rendering it more efficient
        than other strategies, and is also more stable since we sample within
        the complete range of the consituent parts of the CDF uniformly, thus
        ensuring a uniform final distribution.

        Args:
            N (int): the number of samples to return

        Returns:
            tuple(ndarray, ndarray): a tuple containing two numpy ndarrays, the
                first of which are the x coordiantes of the samples, and the
                second of which are the y coordinates.

        """

        # sample angle theta ~ U[0, 2*pi]
        theta = self.rng.norm_sample(lower=0, upper=2 * np.pi, N=N)
        # rho = sqrt(u), where u ~ U[0, 1]
        u = self.rng.norm_sample(lower=0, upper=1, N=N)
        rho = np.sqrt(u)
        # transform rho and theta into Euclidean coordinates
        x = self.r * rho * np.cos(theta)
        y = self.r * rho * np.sin(theta)
        return (x, y)

    def _polar(self, N=1e6):
        """Perform disc sampling using polar coordinate sampling.

        Args:
            N (int): the number of samples to return

        Returns:
            tuple(ndarray, ndarray): a tuple containing two numpy ndarrays, the
                first of which are the x coordiantes of the samples, and the
                second of which are the y coordinates.

        """

        # sample rho, theta ~ U[0, 1]
        rho = self.rng.norm_sample(lower=0, upper=1, N=N)
        theta = self.rng.norm_sample(lower=0, upper=1, N=N)
        # transform rho and theta into Euclidean coordinates
        x = self.r * rho * np.cos(2 * np.pi * theta)
        y = self.r * rho * np.sin(2 * np.pi * theta)
        return (x, y)

    def _rejection(self, N=1e6):
        """Perform disc sampling using rejection sampling.

        Args:
            N (int): the number of samples to return

        Returns:
            tuple(ndarray, ndarray): a tuple containing two numpy ndarrays, the
                first of which are the x coordiantes of the samples, and the
                second of which are the y coordinates.

        """

        # sample (x, y) coordinates from the same U(-r, r) distribution
        samples = np.array(
            [
                list(self.rng.norm_sample(lower=-self.r, upper=self.r, N=2))
                for i in range(N)
            ]
        )
        # calculate the distance from each coordinate to the origin
        norms = np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)
        # accept all those points at least as close to the origin at the disc's radius
        accepted = samples[norms <= self.r]
        return (samples[:, 0], samples[:, 1])

    def sample(self, N=1e6):
        """The main sampling method of the Disc class.

        Args:
            N (int, float): the number of samples to return

        Returns:
            tuple(ndarray, ndarray): a tuple containing two numpy ndarrays, the
                first of which are the x coordiantes of the samples, and the
                second of which are the y coordinates.

        """

        # ensure N is an integer
        if isinstance(N, float):
            N = int(N)
        # define available sampling methods
        sampling_methods = {
            "inverse": self._inverse(N),
            "polar": self._polar(N),
            "rejection": self._rejection(N),
        }
        return sampling_methods.get(self.strategy)
