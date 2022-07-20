"""Markov Chain Monte Carlo sampling module."""

import numpy as np

from pafnuty.samplers.rng import LFG


class HMC:
    """Hamiltonian MCMC class."""

    def __init__(
        self,
        path_len=1,
        step_size=0.25,
        epochs=1_000,
    ):
        self.path_len = path_len
        self.step_size = step_size
        self.epochs = epochs

    def sample(self, dist, momentum):
        """Use Hamiltonian MCMC to sample from a distribution.

        Code for this method is modified from Moore (2020), original accessible
        at https://gist.github.com/jdmoore7/ef28520f1c7663318c5f1174a2564bd1.

        Args:
            dist (pafnuty.Dist): a natively-defined probability distribution
                which we want to sample from.
            momentum (pafnuty.Dist): a natively-defined probability distribution
                of the momentum distribution for the simulation.

        Returns:
            ndarray: trace of MCMC sampling.

        Example:

            >>> import pafnuty as paf
            >>> import pafnuty.samplers as samplers
            >>> # define probability distribution to sample from
            >>> # and the momentum distribution to use
            >>> dist = paf.Normal(mu=0, sigma=1)
            >>> momentum = paf.Normal(mu=0, sigma=1)
            >>> # initialise Hamiltonian Monte Carlo kernel
            >>> mcmc = samplers.HMC()
            >>> # sample from the distribution
            >>> mcmc.sample(dist, momentum)
        """

        if dist.name is not "Normal" or momentum.name is not "Normal":
            raise ValueError(
                "Unfortunately this method currently only supports the Normal "
                + "distribution."
            )
        # setup
        steps = int(self.path_len / self.step_size)
        samples = [momentum.sample().item()]
        # generate samples
        for e in range(self.epochs):
            q0 = np.copy(samples[-1])
            q1 = np.copy(q0)
            p0 = momentum.sample().item()
            p1 = np.copy(p0)
            # ensure dtype
            q0, q1, p0, p1 = float(q0), float(q1), float(p0), float(p1)
            # compute derivative of pdf at q0
            dVdQ = dist.dVdQ(q0).item()

            # leapfrog integration begin
            for s in range(steps):
                p1 += self.step_size * dVdQ / 2
                q1 += self.step_size * p1
                p1 += self.step_size * dVdQ / 2
            # flip momentum for reversibility
            p1 = -1 * p1

            # metropolis acceptance
            q0_nlp = -dist.logpdf(x=q0)
            q1_nlp = -dist.logpdf(x=q1)

            p0_nlp = -momentum.logpdf(x=p0)
            p1_nlp = -momentum.logpdf(x=p1)

            target = q0_nlp - q1_nlp  # P(q1)/P(q0)
            adjustment = p1_nlp - p0_nlp  # P(p1)/P(p0)
            acceptance = target + adjustment

            # sample from U[0, 1) with LFG pseudo-RNG
            rng = LFG()
            event = np.log(rng.norm_sample())
            if event <= acceptance:
                samples.append(q1)
            else:
                samples.append(q0)

        self.trace = np.array(samples)
        return self.trace

    def plot_trace(self):
        """Plot the HMC sampling trace."""

        if self.trace is None:
            raise ValueError(
                "Please execute the sampling before calling any visualisation methods."
            )

        pass

    def plot_trajectory(self):
        """Plot the trajectory of the HMC sampler through the sampling space."""

        if self.trace is None:
            raise ValueError(
                "Please execute the sampling before calling any visualisation methods."
            )
        pass
