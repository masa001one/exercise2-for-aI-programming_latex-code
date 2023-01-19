import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Generate sample data
np.random.seed(1)
X = np.random.uniform(-3, 3, size=(100, 1))
y = np.sin(X) + np.random.normal(0, 0.1, size=(100, 1))

# Define kernel and Gaussian process regressor
kernel = RBF()
gp = GaussianProcessRegressor(kernel=kernel)

# Fit the model to the data
gp.fit(X, y)

# Create the grid for the plot
x_grid = np.linspace(-3, 3, 100)
x_grid = x_grid.reshape(-1, 1)

# Compute the predicted mean and standard deviation
y_mean, y_std = gp.predict(x_grid, return_std=True)

# ...

# Plot the data and the model
fig = plt.figure()  # Figure object creation
gs = GridSpec(1, 1, fig)  # divide fig into 1 row, 1 column
ax = fig.add_subplot(gs[0, 0])  # whole section for model plot

ax.plot(x_grid, np.sin(x_grid), c='black', ls='--', lw=2, label='true')
ax.plot(x_grid, y_mean, c='tab:blue', ls='-', lw=2, label='predictive mean')

# 95% credible interval plot
ax.fill_between(x_grid.ravel(), y_mean.ravel() + 1.96 * y_std, y_mean.ravel() - 1.96 * y_std,
                fc='tab:blue', alpha=0.2, label='95% credible interval')

ax.scatter(
    X,
    y,
    marker='o',
    s=50,
    lw=1.3,
    c='gold',
    ec='black',
    zorder=4,
    label='observed')

next_point = np.array([[1.5]])  # here next_point is defined.
pred, std = gp.predict(next_point, return_std=True)
ax.scatter(
    next_point[0],
    pred,
    marker='x',
    color='black',
    label='Next observation')
ax.axvline(next_point[0], ls=':', lw=1.5, c='tab:green', label='next')

ax.set_title(f'Gaussian Process Regression')
ax.set_ylabel(r'y')
ax.set_xlabel(r'x')
ax.legend(
    ncol=2,
    fontsize=11,
    framealpha=0.4,
    loc='lower left',
    borderaxespad=0.1)

fig.tight_layout()
plt.show()
