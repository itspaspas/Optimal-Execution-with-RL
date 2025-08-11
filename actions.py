import numpy as np

class ActionStrategy:
    def compute(self, env, action):
        raise NotImplementedError


class BaselineAction(ActionStrategy):
    def compute(self, env, action):
        return min(env.constantSharesToSell * action, env.shares_remaining)


class SpreadAction(ActionStrategy):
    def compute(self, env, action):
        spread_factor = env.basp / env.prevPrice
        price_momentum = env.logReturns[-1]
        adjustment = 1 + (spread_factor * price_momentum)
        return env.shares_remaining * action * adjustment


class VolatilityAction(ActionStrategy):
    def compute(self, env, action):
        recent_volatility = np.std(list(env.logReturns))
        vol_adjustment = recent_volatility / env.sigma
        return env.shares_remaining * action * vol_adjustment


class TimeAction(ActionStrategy):
    def compute(self, env, action):
        time_factor = env.timeHorizon / env.num_n
        return env.shares_remaining * action * time_factor


class VolumeAction(ActionStrategy):
    def compute(self, env, action):
        max_daily_participation = 0.01
        shares = env.dtv * max_daily_participation * action
        return min(shares, env.shares_remaining)


class PercentAction(ActionStrategy):
    def compute(self, env, action):
        return min(env.total_shares * action, env.shares_remaining)


class RateAction(ActionStrategy):
    def compute(self, env, action):
        trading_rate = action
        return env.shares_remaining * trading_rate / max(env.timeHorizon, 1)

action_registry = {
    "baseline": BaselineAction(),
    "spread": SpreadAction(),
    "volatility": VolatilityAction(),
    "time": TimeAction(),
    "volume": VolumeAction(),
    "percent": PercentAction(),
    "rate": RateAction(),
}