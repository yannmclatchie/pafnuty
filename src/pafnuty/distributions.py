"""Probability distribution classes for sampling modules."""

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy import special


class Dist:
    """Base probability distribution class."""

    def pdf(x):
        raise NotImplementedError

    def logpdf(x):
        raise NotImplementedError

    def cdf(x):
        raise NotImplementedError

    def invcdf(x):
        raise NotImplementedError

    def dVdQ(x):
        raise NotImplementedError

    def sample(N):
        raise NotImplementedError


class Normal(Dist):
    """Normal distribution class."""

    def __init__(self, mu=0, sigma=1, seed=300416):
        """Initialise the distribution's parameters."""

        self.mu = mu
        self.sigma = sigma
        self.key = random.PRNGKey(seed)
        self.name = "Normal"

    def pdf(self, x):
        """Return the probability distribution at a point x.

        Args:
            x (float, int): the point at which to compute the PDF.

        Returns:
            jax.DeviceArray: a one element sized DeviceArray containing the
                value of the PDF at x.

        """

        return (jnp.exp(-1 * ((x - self.mu) ** 2) / (2 * self.sigma**2))) / (
            self.sigma * jnp.sqrt(2 * jnp.pi)
        )

    def logpdf(self, x):
        """Return the log probability distribution at a point x.

        Args:
            x (float, int): the point at which to compute the log of the PDF.

        Returns:
            jax.DeviceArray: a one element sized DeviceArray containing the
                log of the value of the PDF at x.

        """

        return jnp.log(self.pdf(x))

    def cdf(self, x):
        """Return the cumulative probability distribution up to point x.

        Args:
            x (float, int): the point up to which to compute the CDF.

        Returns:
            jax.DeviceArray: a one element sized DeviceArray containing the
                value of the CDF up to x.

        """

        return 1 / 2 * (1 + special.erf((x - self.mu) / (self.sigma * jnp.sqrt(2))))

    def invcdf(self, x):
        """Return the inverse cumulative probability distribution up to point x.

        Args:
            x (float, int): the point up to which to compute the inverse CDF.

        Returns:
            jax.DeviceArray: a one element sized DeviceArray containing the
                value of the inverse CDF up to x.

        """

        return self.mu + self.sigma * special.erfinv(2 * x - 1) * jnp.sqrt(2)

    def dVdQ(self, x):
        """Return gradient of PDF at a point x with JAX autodiff.

        Args:
            x (float): the point at which to compute the gradient of the PDF.

        Returns:
            jax.DeviceArray: a one element sized DeviceArray containing the
                gradient of the PDF at x.

        """

        if not isinstance(x, float):
            raise ValueError(
                "dVdQ accepts only real or complex-valued inputs, not ints."
            )
        return jax.grad(self.pdf)(x)

    def sample(self, N=1):
        """Sample N data points from the Normal distribution.

        This method leverages the inverse cumulative distribution sampling
        technique to draw its samples using pafnuty's native Pseudo-RNGs.

        Args:
            N (int): the number of samples to draw from the distribution.

        Returns:
            jax.DeviceArray: DeviceArray of N samples from distribution.

        To do:
            * Implement using native pseudo-RNGs, e.g.

                >>> rng = samplers.LFG()
                >>> u = rng.norm_sample(N=N)
                >>> x = self.invcdf(u)
                >>> return x

                There is currently some error with this sampling method producing
                incorrect samples.
        """

        return random.normal(self.key, (N, 1)).flatten() * self.sigma + self.mu
