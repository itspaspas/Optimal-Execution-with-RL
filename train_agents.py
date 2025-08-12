import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Callable, List, Tuple, Any

from syntheticChrissAlmgren import (
    MarketEnvironment,
    TOTAL_SHARES,
    STARTING_PRICE,
    ETA,
    LIQUIDATION_TIME,
    NUM_N,
)
from actions import action_registry
from rewards import (
    PnL,
    CjOeCriterion,
    CjMmCriterion,
    RunningInventoryPenalty,
    NormalizedExecutionReward,
    ExecutionShortfallWithPenaltiesReward,
    ACUtilityReward,
)
from ddpg_agent import Agent as DDPGAgent
from td3_agent import TD3
from sac_agent import SACAgent

# ---------------------------------------------------------------------------
# Registry of reward and action functions
# ---------------------------------------------------------------------------

tau_const = LIQUIDATION_TIME / NUM_N

REWARD_FUNCTIONS: Dict[str, Callable[[], Any]] = {
    "pnl": lambda: PnL(),
    "cj_oe": lambda: CjOeCriterion(),
    "cj_mm": lambda: CjMmCriterion(),
    "running_inventory": lambda: RunningInventoryPenalty(),
    "normalized": lambda: NormalizedExecutionReward(TOTAL_SHARES, STARTING_PRICE),
    "shortfall_penalty": lambda: ExecutionShortfallWithPenaltiesReward(
        P0=STARTING_PRICE,
        alpha=0.1,
        eta=ETA,
        tau=tau_const,
        leftover_penalty=1e-3,
        total_shares=TOTAL_SHARES,
    ),
    "ac_utility": lambda: ACUtilityReward(),
}

ACTION_FUNCTIONS = action_registry

# ---------------------------------------------------------------------------
# Environment and training helpers
# ---------------------------------------------------------------------------

def create_env(reward_name: str = "pnl", action_name: str = "baseline", seed: int = 0,
               price_model: str = "gbm") -> MarketEnvironment:
    """Instantiate the market environment with specified reward and action strategies."""
    reward_fn = REWARD_FUNCTIONS[reward_name]()
    env = MarketEnvironment(
        randomSeed=seed,
        reward_function=reward_fn,
        action_type=action_name,
        price_model=price_model,
    )
    if hasattr(reward_fn, "env"):
        reward_fn.env = env
    env.reset(seed)
    return env

def train_agent(agent, env: MarketEnvironment, n_episodes: int = 10, max_t: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """Generic training loop for an agent in the environment."""
    scores: List[float] = []
    moving_avg: List[float] = []
    shortfalls: List[float] = []
    step_rewards: List[List[float]] = []
    cumulative_rewards: List[List[float]] = []
    window = 10

    for i_episode in range(1, n_episodes + 1):
        state = env.reset(seed)
        env.start_transactions()
        agent.reset()
        episode_reward = 0.0
        steps: List[float] = []
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward_arr, done, info = env.step(float(action[0]))
            reward = float(reward_arr[0])
            agent.step(state, action, reward, next_state, done)
            episode_reward += reward
            steps.append(reward)
            state = next_state
            if done:
                break
        scores.append(episode_reward)
        step_rewards.append(steps)
        cumulative_rewards.append(list(np.cumsum(steps)))
        shortfalls.append(getattr(info, "implementation_shortfall", np.nan))
        moving_avg.append(np.mean(scores[-window:]))

    return {
        "scores": scores,
        "moving_avg": moving_avg,
        "shortfalls": shortfalls,
        "step_rewards": step_rewards,
        "cumulative_rewards": cumulative_rewards,
    }

def run_experiment(algo: str, reward_name: str, action_name: str,
                   n_episodes: int = 10, max_t: int = 1000, seed: int = 0,
                   agent_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
    """Configure environment and agent then train."""
    env = create_env(reward_name, action_name, seed)
    agent_kwargs = agent_kwargs or {}
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()
    if algo.lower() == "ddpg":
        agent = DDPGAgent(state_size, action_size, seed, **agent_kwargs)
    elif algo.lower() == "td3":
        agent = TD3(state_size, action_size, seed, **agent_kwargs)
    elif algo.lower() == "sac":
        agent = SACAgent(state_size, action_size, seed, **agent_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    results = train_agent(agent, env, n_episodes, max_t, seed)
    results["ac_expected_shortfall"] = env.get_AC_expected_shortfall(env.total_shares)
    return results

def hyperparameter_tuning(algo: str, reward_name: str, action_name: str,
                          param_name: str, values: List[Any],
                          n_episodes: int = 5, max_t: int = 1000,
                          seed: int = 0) -> Tuple[Any, List[Tuple[Any, float]]]:
    """Simple grid search for a single hyperparameter."""
    history: List[Tuple[Any, float]] = []
    best_val: Any = None
    best_score = -np.inf
    for val in values:
        agent_kwargs = {param_name: val}
        results = run_experiment(algo, reward_name, action_name,
                                 n_episodes=n_episodes, max_t=max_t,
                                 seed=seed, agent_kwargs=agent_kwargs)
        avg_score = float(np.mean(results["scores"]))
        history.append((val, avg_score))
        if avg_score > best_score:
            best_score = avg_score
            best_val = val
    return best_val, history

def plot_results(results: Dict[str, Any], title: str = "Results") -> None:
    """Plot training statistics from the results dictionary."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    episodes = np.arange(1, len(results["scores"]) + 1)

    axes[0, 0].plot(episodes, results["shortfalls"], label="Shortfall")
    axes[0, 0].set_title("Shortfall per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Shortfall")

    axes[0, 1].plot(episodes, results["scores"], label="Reward")
    axes[0, 1].plot(episodes, results["moving_avg"], label="Moving Avg")
    axes[0, 1].set_title("Reward per Episode")
    axes[0, 1].legend()

    ac_shortfall = results.get("ac_expected_shortfall", None)
    axes[1, 0].plot(episodes, results["shortfalls"], label="RL")
    if ac_shortfall is not None:
        axes[1, 0].hlines(ac_shortfall, 1, len(episodes), colors="r", linestyles="--", label="AC")
    axes[1, 0].set_title("Shortfall Comparison")
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, np.cumsum(results["scores"]), label="Cumulative Reward")
    axes[1, 1].plot(episodes, results["scores"], label="Reward per Episode")
    axes[1, 1].set_title("Cumulative Reward")
    axes[1, 1].legend()

    if results["step_rewards"]:
        axes[2, 0].plot(results["step_rewards"][-1])
        axes[2, 0].set_title("Step Rewards (Last Episode)")
        axes[2, 0].set_xlabel("Step")
        axes[2, 0].set_ylabel("Reward")

    axes[2, 1].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run a small demonstration when executed directly
    demo = run_experiment("ddpg", "pnl", "baseline", n_episodes=1, max_t=1)
    print("Demo keys:", list(demo.keys()))
