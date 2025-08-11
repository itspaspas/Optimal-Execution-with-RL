CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3
BID_INDEX = 0
ASK_INDEX = 1

import abc
from typing import Union
import numpy as np

class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass


class PnL(RewardFunction):
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        assert len(current_state.shape) > 1, "Reward functions must be calculated on state matrices."
        current_market_value = (
            current_state[:, CASH_INDEX] + current_state[:, INVENTORY_INDEX] * current_state[:, ASSET_PRICE_INDEX]
        )
        next_market_value = (
            next_state[:, CASH_INDEX] + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX]
        )
        return next_market_value - current_market_value

    def reset(self, initial_state: np.ndarray):
        pass


class CjOeCriterion(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        terminal_time: float = 1.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent
        self.terminal_time = terminal_time
        self.initial_inventory = None
        self.episode_length = None

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - dt
            * self.terminal_inventory_aversion
            * (
                self.inventory_exponent
                * np.squeeze(action)
                * (current_state[:, INVENTORY_INDEX]) ** (self.inventory_exponent - 1)
                + self.initial_inventory**self.inventory_exponent * self.episode_length
            )
        )

    def reset(self, initial_state: np.ndarray):
        self.initial_inventory = initial_state[:, INVENTORY_INDEX]
        self.episode_length = self.terminal_time - initial_state[:, TIME_INDEX]


class CjMmCriterion(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        terminal_time: float = 1.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent
        self.terminal_time = terminal_time
        self.initial_inventory = None
        self.episode_length = None

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - self.terminal_inventory_aversion
            * (
                next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
                - current_state[:, INVENTORY_INDEX] ** self.inventory_exponent
                + dt / self.episode_length * self.initial_inventory**self.inventory_exponent
            )
        )

    def reset(self, initial_state: np.ndarray):
        self.initial_inventory = initial_state[:, INVENTORY_INDEX]
        self.episode_length = self.terminal_time - initial_state[:, TIME_INDEX]


class RunningInventoryPenalty(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - self.terminal_inventory_aversion
            * int(is_terminal_step)
            * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
        )

    def reset(self, initial_state: np.ndarray):
        pass

CjCriterion = RunningInventoryPenalty

class ExponentialUtility(RewardFunction):
    def __init__(self, risk_aversion: float = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        return (
            -np.exp(
                -self.risk_aversion
                * (next_state[:, CASH_INDEX] + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX])
            )
            if is_terminal_step
            else 0
        )

    def reset(self, initial_state: np.ndarray):
        pass

class NormalizedExecutionReward(RewardFunction):
    def __init__(self, total_shares: float, P0: float, price_index: int = 0):
        self.total_shares = float(total_shares)
        self.P0 = float(P0)
        self.price_index = price_index

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal_step: bool = False,
    ) -> float:
        flat_state = np.asarray(next_state).ravel()

        Q_t = float(action[0])
        exec_price = float(flat_state[self.price_index])

        r_bar = Q_t * exec_price

        return (r_bar - Q_t * self.P0) / self.total_shares

    def reset(self, initial_state: np.ndarray):
        # Stateless
        pass

class ExecutionShortfallWithPenaltiesReward(RewardFunction):
    def __init__(
        self,
        P0: float,
        alpha: float,
        eta: float,
        tau: float,
        leftover_penalty: float,
        total_shares: float,
        *,
        price_index: int = 0,
        shares_index: int = 1,
        temp_impact_index: int = 2,
    ):
        self.P0 = float(P0)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.tau = float(tau)
        self.leftover_penalty = float(leftover_penalty)
        self.total_shares = float(total_shares)

        self.price_index = price_index
        self.shares_index = shares_index
        self.temp_impact_index = temp_impact_index

        self._beta = 1e-6
        self._delta = 1e-3

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal_step: bool = False,
    ) -> float:
        flat_state = np.asarray(next_state).ravel()

        Q_t = float(action[0])

        P_t = float(flat_state[self.price_index])
        shares_remaining = float(flat_state[self.shares_index])
        current_temp_impact = float(flat_state[self.temp_impact_index])

        reward = Q_t * (self.P0 - P_t)

        d_t = abs(current_temp_impact) / (self.eta / self.tau)
        reward -= self.alpha * d_t

        if is_terminal_step and shares_remaining > 0:
            reward -= self.leftover_penalty * shares_remaining

        reward -= self._beta * (Q_t ** 2)
        reward -= self._delta * (shares_remaining / self.total_shares)

        frac = Q_t / self.total_shares
        reward -= 1e5 * (np.exp(5.0 * frac) - 1.0)

        return reward

    def reset(self, initial_state: np.ndarray):
        # Stateless
        pass

class ACUtilityReward(RewardFunction):
    def __init__(self):
        self.prevUtility: float = None
        self.env = None

    def reset(self, initial_state: np.ndarray):
        self.prevUtility = self.env.compute_AC_utility(self.env.total_shares)

    def calculate(
    self,
    current_state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    is_terminal_step: bool = False
    ) -> float:
        frac = float(next_state[0, -1])
        shares_rem = frac * self.env.total_shares
        currentUtility = self.env.compute_AC_utility(shares_rem)
        reward = 0.0
        if self.prevUtility not in (None, 0.0):
            reward = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
        self.prevUtility = currentUtility
        return reward

class TerminalOnlyReward(RewardFunction):
    def __init__(self, base_reward: RewardFunction):
        self.base = CjOeCriterion

    def reset(self, initial_state):
        self.base.reset(initial_state)

    def calculate(self, curr, act, nxt, is_terminal_step=False):
        r = self.base.calculate(curr, act, nxt, is_terminal_step)
        return r if is_terminal_step else 0.0