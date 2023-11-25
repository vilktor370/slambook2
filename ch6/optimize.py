from scipy.optimize import minimize, rosen_der
import numpy as np

x0 = np.array([0, 0])
func = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
bounds = np.array([[0, None], [0, None]])
cons = (
    {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
    {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
    {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
)
jacobina = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1]- 2.5)])
res = minimize(func, x0, bounds=bounds, jac=jacobina)
print(res)
