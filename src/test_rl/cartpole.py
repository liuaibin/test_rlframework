from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--stop-reward", type=float, default=450.0)
parser.add_argument("--stop-timesteps", type=int, default=300000)
args = parser.parse_args()

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        lr=0.0003,
        num_epochs=6,
        vf_loss_coeff=0.01,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="linear",
            vf_share_layers=True,
        ),
    )
)


if __name__ == "__main__":
    # Build the algorithm
    algo = config.build()

    # Run training until stop condition is met
    while True:
        result = algo.train()
        print("=========:",result)

    algo.stop()