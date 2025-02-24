# FlexiPPO
pip install gymnasium stable-baselines3 torch numpy
pip install stable-baselines3==2.0.0
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np
import os
from collections import deque

# تنظیمات پایه
config_base = {
    "env_name": "CartPole-v1",
    "num_envs": 4,
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "target_reward": 490,
    "eval_freq": 10000,
    "log_dir": "./ppo_cartpole_log/",
    "model_name": "flexippo_best_model",
    "normalize_env": True,
    "seed": 42
}

# تابع ساخت محیط
def make_env(env_name: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed)
        return env
    return _init

# Callback برای لاگ کردن متریک‌ها
class MetricLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            "explained_variance": 0,
            "value_loss": 0,
            "approx_kl": 0,
            "entropy_loss": 0
        }
        self.reward_history = deque(maxlen=5)

    def _on_step(self) -> bool:
        try:
            if "train/explained_variance" in self.model.logger.name_to_value:
                self.metrics["explained_variance"] = self.model.logger.name_to_value["train/explained_variance"]
            if "train/value_loss" in self.model.logger.name_to_value:
                self.metrics["value_loss"] = self.model.logger.name_to_value["train/value_loss"]
            if "train/approx_kl" in self.model.logger.name_to_value:
                self.metrics["approx_kl"] = self.model.logger.name_to_value["train/approx_kl"]
            if "train/entropy_loss" in self.model.logger.name_to_value:
                self.metrics["entropy_loss"] = self.model.logger.name_to_value["train/entropy_loss"]
        except Exception as e:
            print(f"Error in logging metrics: {e}")
        return True

# تابع تحلیل و تنظیم خودکار
def analyze_and_adjust(model, eval_callback, config, metric_logger):
    mean_reward = eval_callback.last_mean_reward
    explained_variance = metric_logger.metrics["explained_variance"]
    value_loss = metric_logger.metrics["value_loss"]
    approx_kl = metric_logger.metrics["approx_kl"]
    entropy_loss = metric_logger.metrics["entropy_loss"]

    print(f"FlexiPPO Analyzing: mean_reward={mean_reward}, explained_variance={explained_variance}, "
          f"value_loss={value_loss}, approx_kl={approx_kl}, entropy_loss={entropy_loss}")

    adjustments_made = False
    metric_logger.reward_history.append(mean_reward)

    if len(metric_logger.reward_history) >= 2:
        reward_change = abs(mean_reward - metric_logger.reward_history[-2])
        if reward_change > 5:
            print(f"FlexiPPO: Sudden reward change detected ({reward_change}). Resetting parameters.")
            config["learning_rate"] = config_base["learning_rate"]
            config["ent_coef"] = config_base["ent_coef"]
            config["clip_range"] = config_base["clip_range"]
            model.learning_rate = config["learning_rate"]
            model.ent_coef = config["ent_coef"]
            model.clip_range = lambda _: config["clip_range"]
            adjustments_made = True

    if not adjustments_made:
        if explained_variance < 0.5 and value_loss > 0.3:
            new_lr = min(config["learning_rate"] * 1.2, 0.001)
            config["learning_rate"] = new_lr
            model.learning_rate = new_lr
            adjustments_made = True
            print(f"FlexiPPO: Adjusted learning_rate to {new_lr} due to low explained_variance and high value_loss")
        elif explained_variance > 0.9 and value_loss < 0.01:
            new_lr = max(config["learning_rate"] * 0.8, 0.0001)
            config["learning_rate"] = new_lr
            model.learning_rate = new_lr
            adjustments_made = True
            print(f"FlexiPPO: Reduced learning_rate to {new_lr} due to high convergence")

        if approx_kl < 0.005 or entropy_loss > -0.1:
            new_ent_coef = min(config["ent_coef"] * 1.5, 0.1)
            config["ent_coef"] = new_ent_coef
            model.ent_coef = new_ent_coef
            adjustments_made = True
            print(f"FlexiPPO: Adjusted ent_coef to {new_ent_coef} due to low approx_kl or insufficient exploration")

        if mean_reward < 10 and model.num_timesteps > 40000:
            new_clip_range = max(config["clip_range"] * 0.9, 0.1)
            config["clip_range"] = new_clip_range
            model.clip_range = lambda _: new_clip_range
            adjustments_made = True
            print(f"FlexiPPO: Adjusted clip_range to {new_clip_range} due to slow reward growth")

    if not adjustments_made:
        print("FlexiPPO: No adjustments needed at this step")

# Callback سفارشی برای تحلیل و تنظیم
class AdaptiveCallback(EvalCallback):
    def __init__(self, env, config, metric_logger, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.config = config
        self.metric_logger = metric_logger

    def _on_step(self):
        try:
            result = super()._on_step()
            if self.n_calls % self.eval_freq == 0:
                analyze_and_adjust(self.model, self, self.config, self.metric_logger)
            return result
        except Exception as e:
            print(f"FlexiPPO: Error during evaluation: {e}. Restarting evaluation...")
            self.model.env.reset()
            return True

# آموزش FlexiPPO
if __name__ == "__main__":
    config = config_base.copy()
    experiment_name = f"FlexiPPO_LR_{config['learning_rate']}_ENT_{config['ent_coef']}"
    experiment_log_dir = os.path.join(config["log_dir"], experiment_name)
    os.makedirs(experiment_log_dir, exist_ok=True)
    print(f"\n======================== Starting Experiment: {experiment_name} =========================")

    np.random.seed(config["seed"])
    envs = SubprocVecEnv([make_env(config["env_name"], i, seed=config["seed"]) for i in range(config["num_envs"])])
    if config["normalize_env"]:
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        verbose=1,
        tensorboard_log=experiment_log_dir,
        seed=config["seed"]
    )

    metric_logger = MetricLoggerCallback()
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=config["target_reward"], verbose=1)
    eval_callback = AdaptiveCallback(
        envs,
        config,
        metric_logger,
        best_model_save_path=experiment_log_dir,
        log_path=experiment_log_dir,
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[metric_logger, eval_callback]
        )
    except Exception as e:
        print(f"FlexiPPO: Critical error during training: {e}. Saving model and exiting...")
        model.save(os.path.join(experiment_log_dir, "flexippo_error_model"))

    model.save(os.path.join(experiment_log_dir, "flexippo_final_model"))
    print(f"\n======================== Experiment {experiment_name} Finished =========================")
    
