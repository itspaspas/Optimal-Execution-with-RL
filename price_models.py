import numpy as np
import math
import random
import collections
from typing import Deque

def abm_price(prev_price: float, single_step_variance: float, tau: float) -> float:
    return prev_price + np.sqrt(single_step_variance * tau) * random.normalvariate(0, 1)


def ar_l_price(prev_impacted_price: float,
               alphas: np.ndarray,
               lag_coeffs: np.ndarray,
               a_deque: collections.deque,
               returns_deque: collections.deque) -> float:
    sigma_sq = alphas[0]
    for j in range(1, len(alphas)):
        sigma_sq += alphas[j] * (a_deque[-j] ** 2)
    sigma_sq = min(sigma_sq, 200)

    new_a = np.sqrt(sigma_sq) * random.normalvariate(0, 1)
    a_deque.append(new_a)
    a_deque.popleft()

    new_return = new_a
    for j in range(len(lag_coeffs)):
        new_return += lag_coeffs[j] * returns_deque[-j-1]
    returns_deque.append(new_return)
    returns_deque.popleft()

    return prev_impacted_price * np.exp(0.0001 * new_return)


import numpy as np

def gbm_price(prev_price: float,
              dt: float,
              sigma: float,
              mu: float
             ) -> float:
    dw = np.random.normal(loc=0.0, scale=np.sqrt(dt) * sigma)

    drift = (mu - 0.5 * sigma**2) * dt
    log_ret = drift + dw

    return prev_price * np.exp(log_ret)