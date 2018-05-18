# Many Thanks to Christopher Strelioff for this code.
#http://chrisstrelioff.ws/sandbox/2014/12/11/inferring_probabilities_with_a_beta_prior_a_third_example_of_bayesian_calculations.html
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass

class likelihood:
    def __init__(self, data):
        """Likelihood for binary data."""
        self.counts = {s:0 for s in ['0', '1']}
        self._process_data(data)

    def _process_data(self, data):
        """Process data."""
        temp = [str(x) for x in data]
        for s in ['0', '1']:
            self.counts[s] = temp.count(s)

        if len(temp) != sum(self.counts.values()):
            raise Exception("Passed data is not all 0`s and 1`s!")

    def _process_probabilities(self, p0):
        """Process probabilities."""
        n0 = self.counts['0']
        n1 = self.counts['1']

        if p0 != 0 and p0 != 1:
            # typical case
            logpr_data = n0*np.log(p0) + \
                         n1*np.log(1.-p0)
            pr_data = np.exp(logpr_data)
        elif p0 == 0 and n0 != 0:
            # p0 can't be 0 if n0 is not 0
            logpr_data = -np.inf
            pr_data = np.exp(logpr_data)
        elif p0 == 0 and n0 == 0:
            # data consistent with p0=0
            logpr_data = n1*np.log(1.-p0)
            pr_data = np.exp(logpr_data)
        elif p0 == 1 and n1 != 0:
            # p0 can't be 1 if n1 is not 0
            logpr_data = -np.inf
            pr_data = np.exp(logpr_data)
        elif p0 == 1 and n1 == 0:
            # data consistent with p0=1
            logpr_data = n0*np.log(p0)
            pr_data = np.exp(logpr_data)

        return pr_data, logpr_data

    def prob(self, p0):
        """Get probability of data."""
        pr_data, _ = self._process_probabilities(p0)

        return pr_data

    def log_prob(self, p0):
   
        """Get log of probability of data."""
        _, logpr_data = self._process_probabilities(p0)

        return logpr_data
class prior:
    def __init__(self, alpha0=1, alpha1=1):
        """Beta prior for binary data."""

        self.a0 = alpha0
        self.a1 = alpha1
        self.p0rv = beta(self.a0, self.a1)

    def interval(self, prob):
        """End points for region of pdf containing `prob` of the
        pdf-- this uses the cdf and inverse.

        Ex: interval(0.95)
        """

        return self.p0rv.interval(prob)

    def mean(self):
        """Returns prior mean."""

        return self.p0rv.mean()

    def pdf(self, p0):
        """Probability density at p0."""

        return self.p0rv.pdf(p0)

    def plot(self):
        """A plot showing mean and 95% credible interval."""

        fig, ax = plt.subplots(1, 1)
        x = np.arange(0., 1., 0.01)

        # get prior mean p0
        mean = self.mean()

        # get low/high pts containg 95% probability
        low_p0, high_p0 = self.interval(0.95)
        x_prob = np.arange(low_p0, high_p0, 0.01)

        # plot pdf
        ax.plot(x, self.pdf(x), 'r-')

        # fill 95% region
        ax.fill_between(x_prob, 0, self.pdf(x_prob),
                        color='red', alpha='0.2' )

        # mean
        ax.stem([mean], [self.pdf(mean)], linefmt='r-',
                markerfmt='ro', basefmt='w-')

        ax.set_xlabel('Probability of Zero')
        ax.set_ylabel('Prior PDF')
        ax.set_ylim(0., 1.1*np.max(self.pdf(x)))

        plt.show()
        
class posterior:
    def __init__(self, data, prior):
        """The posterior.

        data: a data sample as list
        prior: an instance of the beta prior class
        """
        self.likelihood = likelihood(data)
        self.prior = prior

        self._process_posterior()

    def _process_posterior(self):
        """Process the posterior using passed data and prior."""

        # extract n0, n1, a0, a1 from likelihood and prior
        self.n0 = self.likelihood.counts['0']
        self.n1 = self.likelihood.counts['1']
        self.a0 = self.prior.a0
        self.a1 = self.prior.a1

        self.p0rv = beta(self.a0 + self.n0,
                         self.a1 + self.n1)

    def interval(self, prob):
        """End points for region of pdf containing `prob` of the
        pdf.

        Ex: interval(0.95)
        """

        return self.p0rv.interval(prob)

    def mean(self):
        """Returns posterior mean."""

        return self.p0rv.mean()

    def pdf(self, p0):
        """Probability density at p0."""

        return self.p0rv.pdf(p0)

    def plot(self):
        """A plot showing prior, likelihood and posterior."""

        f, ax= plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        x = np.arange(0., 1., 0.01)

        ## Prior
        # get prior mean p0
        pri_mean = self.prior.mean()

        # get low/high pts containg 95% probability
        pri_low_p0, pri_high_p0 = self.prior.interval(0.95)
        pri_x_prob = np.arange(pri_low_p0, pri_high_p0, 0.01)

        # plot pdf
        ax[0].plot(x, self.prior.pdf(x), 'r-')

        # fill 95% region
        ax[0].fill_between(pri_x_prob, 0, self.prior.pdf(pri_x_prob),
                           color='red', alpha='0.2' )

        # mean
        ax[0].stem([pri_mean], [self.prior.pdf(pri_mean)],
                   linefmt='r-', markerfmt='ro',
                   basefmt='w-')

        ax[0].set_ylabel('Prior PDF')
        ax[0].set_ylim(0., 1.1*np.max(self.prior.pdf(x)))

        ## Likelihood
        # plot likelihood
        lik = [self.likelihood.prob(xi) for xi in x]
        ax[1].plot(x, lik, 'k-')
        ax[1].set_ylabel('Likelihood')

        ## Posterior
        # get posterior mean p0
        post_mean = self.mean()

        # get low/high pts containg 95% probability
        post_low_p0, post_high_p0 = self.interval(0.95)
        post_x_prob = np.arange(post_low_p0, post_high_p0, 0.01)

        # plot pdf
        ax[2].plot(x, self.pdf(x), 'b-')

        # fill 95% region
        ax[2].fill_between(post_x_prob, 0, self.pdf(post_x_prob),
                           color='blue', alpha='0.2' )

        # mean
        ax[2].stem([post_mean], [self.pdf(post_mean)],
                   linefmt='b-', markerfmt='bo',
                   basefmt='w-')

        ax[2].set_xlabel('Probability of Zero')
        ax[2].set_ylabel('Posterior PDF')
        ax[2].set_ylim(0., 1.1*np.max(self.pdf(x)))

        plt.show()   
