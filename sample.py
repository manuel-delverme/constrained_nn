import network
import numpy as np


x = np.linspace(-2, 2, 100)

network.smooth_epsilon_insensitive()

h = x1_hat - x1_target

# eps_h = h.abs() - config.constr_margin
eps_h = smooth_epsilon_insensitive(h, config.constr_margin)
# eps_defect = torch.relu(eps_h)

if isinstance(config.chance_constraint, float):
    broken_constr_prob = torch.tanh(eps_h.abs()).mean()
    prob_defect = broken_constr_prob - config.chance_constraint
    defect = prob_defect.repeat(eps_h.shape)
