""""Learn about Pymc3 and Bayesian statistics."""
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from scipy.optimize import minimize

#### stuff in the book

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

def first_model(alpha=1, beta=1, y_real=np.random.binomial(n=1, p=0.5, size=250)):
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=alpha, beta=beta)
        y_obs = pm.Bernoulli("y_observed", p=theta, observed=y_real)

        # Sample from the posterior
        idata = pm.sample(1000, return_inferencedata=True)

        # Sample from the prior predictive
        prior_predictive = pm.sample_prior_predictive(samples=250, model=model)

        # Sample from the posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(idata)
        for group in posterior_predictive:
            if group not in idata.groups():
                idata.add_groups({group: posterior_predictive[group]})
        for group in prior_predictive:
            if group not in idata.groups():
                idata.add_groups({group: prior_predictive[group]})
    return idata


# y_real = np.random.binomial(n=1, p=0.5, size=500)
# bayz_data = first_model(alpha=1, beta=1, y_real=y_real)


def posterior_plot(data):
    posterior_predictive_samples = data.posterior_predictive["y_pred"].values.flatten()

    # Calculate the mean of the posterior samples for 'theta' if it's directly related to 'y_pred'
    # Otherwise, calculate the expected 'y_pred' from the 'theta' samples
    posterior_mean = np.mean(data.posterior["theta"].values)

    # Number of bins for the histogram
    bins = len(np.unique(posterior_predictive_samples))  # or choose your own number of bins

    # Create the histogram for posterior predictive samples
    plt.hist(posterior_predictive_samples, bins=bins, alpha=0.5, label='posterior predictive')

    # Add a vertical line for the posterior mean
    plt.axvline(x=posterior_mean, color='k', linestyle='--', label='posterior mean')

    plt.xlabel('number of success')
    plt.ylabel('frequency')
    plt.legend()
    plt.title('Posterior Predictive with Posterior Mean')
    plt.show()

def plot_arviz(data):
    az.plot_dist(data.prior["theta"])
    plt.title('Prior Distribution of Theta')
    plt.show()
    for methodname in ['prior', 'posterior']:
        az.plot_ppc(data, group=methodname)
        plt.title(f'{methodname.capitalize()} Checks')
        plt.show()

    az.plot_posterior(data, var_names=["theta"])
    plt.title('Posterior Distribution of Theta')
    plt.show()
    az.plot_trace(data, var_names=["theta"])
    plt.title('trace plot')
    plt.show()

# plot_arviz()
# *************************************************************************************************
# exercises chapter 1
answer1 = """Physics based models, such as in agricultural or crop modeling are different from statistical models. They simplify a complex
process so that it can be understood and predicted. They are based on physical laws and principles. They can simplify too much and make poor
predictions if this is the case. They depend on the quality of the data being fed into them. Very vulnerable to errors in the coding as it is
very easy to make mistakes when hand coding lots of parameter calculations like in physics models."""

answer2 = """1) P(theta | y) = P(y | theta) * P(theta) / P(y), 
2) prior predictive distribution P(y),
3) likelihood P(y | theta), 
4) Posterior P(theta | y), 
5) prior

"""

answer3 = """P(Sunny | July 9th 1816)"""

answer4 = """P(is pope | human) vs P(is human | pope), clearly the former is like 1 / 7 billion and the latter is 100%. Obviously if the pope could
be any of God's creatures, then the second probabliity would be very different and would even change the first one(make it even more unlikely!)
"""

answer5_answer6 = """run functions below"""
from scipy import stats

def shop_poisson(mean_rate=3, sample_size=1000):
    # Returns a Poisson distribution object
    return stats.poisson(mean_rate).rvs(sample_size)

def weight_dogs_uniform(minweight=5, maxweight=20, sample_size=1000):
    # Returns a Uniform distribution object
    return stats.uniform(minweight, maxweight - minweight).rvs(sample_size)

def height_elephants_normal(mu_elephants=5000, sigma_elephants=500, sample_size=1000):
    # Returns a Normal distribution object
    return stats.norm(mu_elephants, sigma_elephants).rvs(sample_size)

def height_humans_skewnorm(alpha_humans=0.5, mu_humans=3, sigma_humans=0.5, sample_size=1000):
    # Returns a Skew Normal distribution object
    return stats.skewnorm(alpha_humans, mu_humans, sigma_humans).rvs(sample_size)

shop = shop_poisson(mean_rate=10, sample_size=100)
weight_dog = weight_dogs_uniform(minweight=4, maxweight=45, sample_size=100)
weight_elephant = height_elephants_normal(mu_elephants=7000, sigma_elephants=700, sample_size=100)
height_human = height_humans_skewnorm(alpha_humans=0.3, mu_humans=1.6, sigma_humans=0.4, sample_size=100)

def plot_answer6_answer7(shop, weight_dog, weight_elephant, height_human):
    seaborn.histplot(shop, kde=True)
    plt.title('Shop Poisson Distribution')
    plt.show()
    seaborn.histplot(weight_dog, kde=True)
    plt.title('Weight of Dogs')
    plt.show()
    seaborn.histplot(weight_elephant, kde=True)
    plt.title('Weight of Elephants')
    plt.show()
    seaborn.histplot(height_human, kde=True)
    plt.title('Height of Humans')
    plt.show()

# plot_answer6_answer7(shop, weight_dog, weight_elephant, height_human)

### *************************************************************************************************

answer7 = """Computer Priros Beta(0.5, 0.5), and others"""

def beta_prior_computer(alpha=0.5, beta=0.5, samples=1000):
    return stats.beta(alpha, beta).rvs(samples)

p1 = beta_prior_computer(alpha=0.5, beta=0.5, samples=200)
p2 = beta_prior_computer(alpha=1, beta=1, samples=200)
p3 = beta_prior_computer(alpha=1, beta=4, samples=200)

def show_beta_priors(p1, p2, p3):
    seaborn.histplot(p1, kde=True)
    plt.title('Beta Prior 0.5, 0.5')
    plt.show()
    seaborn.histplot(p2, kde=True)
    plt.title('Beta Prior 1, 1')
    plt.show()
    seaborn.histplot(p3, kde=True)
    plt.title('Beta Prior 1, 4')
    plt.show()

# show_beta_priors(p1, p2, p3)

#### *************************************************************************************************

answer8 = """"""

def update_beta_binomial(n_trials=[0, 1, 2, 3, 12, 180], success = [0, 1, 1, 1, 6, 59], beta_params=[(0.5, 0.5), (1, 1), (10, 10)]):
    _, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=True, sharex=True)
    axes = np.ravel(axes)
    data = zip(n_trials, success)
    theta = np.linspace(0, 1, 1500)
    cmap = plt.get_cmap('viridis')

    for idx, (N, y) in enumerate(data):
        s_n = ("s" if (N > 1) else "")
        for jdx, (a_prior, b_prior) in enumerate(beta_params):
            p_theta_given_y = stats.beta.pdf(theta, a_prior + y, b_prior + N - y)

            # Correctly assign the color from the colormap using the j index
            color = cmap(jdx / len(beta_params))
            axes[idx].plot(theta, p_theta_given_y, lw=4, color=color)
            axes[idx].set_yticks([])
            axes[idx].set_ylim(0, 12)
            if N > 0:  # Avoid division by zero
                axes[idx].plot(y / N, 0, color="k", marker="o", ms=12)
            axes[idx].set_title(f"{N:4d} trial{s_n}, {y:4d} success")

    # Set common labels
    plt.suptitle('Beta Posterior Distributions')
    plt.xlabel('theta')
    plt.ylabel('Density')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust top to accommodate suptitle
    plt.show()

# update_beta_binomial(n_trials=[0, 1, 2, 3, 12, 180], success = [0, 1, 1, 1, 6, 59], beta_params=[(0.5, 0.5), (1, 1), (1, 2)])
# update_beta_binomial(n_trials=[0, 4, 50, 400, 1000, 6800], success = [0, 3, 20, 200, 600, 3000], beta_params=[(0.5, 0.5), (1, 1), (1, 2)])

## *********************************************************************
answer9 = """Some constraints I can use"""

from scipy.stats import entropy


def entropy_constraints():
    cmap = plt.get_cmap('viridis')
    cons = [[{"type": "eq", "fun": lambda x: np.sum(x) - 1}],
            [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
             {"type": "eq", "fun": lambda x: 1.5 - np.sum(x * np.arange(1, 7))}],
            [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
             {"type": "eq", "fun": lambda x: np.sum(x[[2, 3]]) - 0.8}],
            [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
             {"type": "eq", "fun": lambda x: np.sum(x[[2, 3]]) - 0.2},
             {"type": "eq", "fun": lambda x: np.sum(x[[4, 5]]) - 0.2}]
            ]

    max_ent = []
    for i, c in enumerate(cons):
        color = cmap(i/len(cons))
        val = minimize(lambda x: -entropy(x), x0=[1 / 6] * 6, bounds=[(0., 1.)] * 6,
                       constraints=c)['x']
        max_ent.append(entropy(val))
        plt.plot(np.arange(1, 7), val, 'o--', color=color, lw=2.5)
    plt.legend([f"Constraint {i+1}" for i in range(len(cons))])
    plt.xlabel("$t$")
    plt.ylabel("$p(t)$")
    plt.show()


# entropy_constraints()
# *************************************************************************************************

answer_10 = 'Skipped'

# *************************************************************************************************

answer_11 = """For adult blue whales a halfnorm(mu, sigma=200kg) is weakly informative, as it seems to be plausible.
For adult humans, the same distribution would be uninformative, a max entropy distirbution would be better.
For mice, the same as for humans.
If the prior makes not sense, then it is not informative!
"""

# *************************************************************************************************

answer_12 = """ SKIP"""
# *************************************************************************************************

answer_13 = """
1) distro of values before data = prior distribution
2) distro of values we think we could observe = prior predictive distribution
3) ?
4) predict we observe after using model = posterior predictive distribution
5) numerical summaries = arviz?
6) plot data = arviz?
"""
# *************************************************************************************************

answer_14 = """SKIP"""
# *************************************************************************************************

answer_15 = """
Y ~ N(mu, sigma)
mu ~ N(0, 1)
sigma ~ HN(1)

bayes = P(mu, sigma | Y) = P(Y | mu, sigma) * P(mu) * P(sigma) / P(Y)

likelihood = P(Y | mu, sigma)
posterior = P(mu, sigma | Y)
prior = P(mu) * P(sigma) (?)
"""

# *************************************************************************************************
answer_16 = """SKIP"""

# *************************************************************************************************

answer_17 = """
P(loaded | H) = P(H | loaded) * P(loaded) / P(H)
P(H | loaded) = 1
P(loaded) = 0.5
P(H) = P(H | loaded) * P(loaded) + P(H | fair) * P(fair) = 0.5 * 1 + 0.5 * 0.5 = 0.75

so P(loaded | H) = 1 * 0.5 / 0.75 = 2/3
"""

# *************************************************************************************************

answer_18 = """SEE BELOW"""


print('some code here')
print('some code here')