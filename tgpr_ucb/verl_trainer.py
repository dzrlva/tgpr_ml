import os
import yaml
import ray
from verl import DataProto
from verl.trainer.main_ppo import main_task
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.reward_manager import RewardManager

from tgpr_ucb.data.ds1000 import DS1000Dataset
from tgpr_ucb.data.corpus import ValidatedCorpus
from tgpr_ucb.reward.tgpr_reward import TGPRCustomReward


class TGPRRewardManager(RewardManager):
    def __init__(self, config):
        super().__init__()
        self.config = config
        corpus = ValidatedCorpus(config["reward"]["corpus_path"])
        self.reward_fn = TGPRCustomReward(config, corpus)
        self.group_size = config["training"]["group_size"]

    def verify(self, data):
        prompts = data.non_tensor_batch.get("prompt", [])
        responses = data.batch["responses"]
        rewards = []

        for i in range(len(prompts)):
            group_codes = []
            for j in range(self.group_size):
                idx = i * self.group_size + j
                if idx < len(responses):
                    code = responses[idx]
                    group_codes.append(code)

            for j in range(self.group_size):
                idx = i * self.group_size + j
                if idx < len(responses):
                    code = responses[idx]
                    reward = self.reward_fn.compute(code, group_codes)
                    rewards.append(reward)

        import torch
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        data.batch["rewards"] = reward_tensor
        return data


class TGPRVerlTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.reward_manager = TGPRRewardManager(self.config)

    def train(self):
        ray.init()

        config = self.config
        data_path = config["data"]["train_path"]

        dataset = RLHFDataset(
            data_path,
            prompt_key=config["data"]["prompt_key"],
            max_prompt_length=config["training"]["max_prompt_len"],
            filter_prompts=True,
            cache_dir=".cache"
        )

        trainer_config = {
            "actor_rollout_ref": {
                "model": {"path": config["model"]["name"]},
                "actor": {
                    "optim": {"lr": config["training"]["lr"]},
                    "ppo_mini_batch_size": config["training"]["batch_size"] * config["training"]["group_size"],
                    "ppo_micro_batch_size_per_gpu": 1,
                    "clip_ratio": 0.2,
                    "use_lora": config["model"].get("use_lora", False),
                    "lora_config": {
                        "r": config["model"].get("lora_r", 64),
                        "alpha": config["model"].get("lora_alpha", 16),
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    } if config["model"].get("use_lora") else None
                },
                "rollout": {
                    "name": config["rollout"]["backend"],
                    "temperature": config["rollout"]["temperature"],
                    "top_p": config["rollout"]["top_p"],
                    "max_tokens": config["training"]["max_gen_len"],
                    "tensor_model_parallel_size": config["rollout"]["tensor_parallel"]
                }
            },
            "reward_model": {
                "custom": True,
                "reward_manager": self.reward_manager
            },
            "data": {
                "train_files": data_path,
                "prompt_key": config["data"]["prompt_key"],
                "max_prompt_length": config["training"]["max_prompt_len"],
                "max_response_length": config["training"]["max_gen_len"]
            },
            "trainer": {
                "total_epochs": config["training"]["epochs"],
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "project_name": "tgpr-ucb",
                "experiment_name": config["model"]["name"].replace("/", "_")
            }
        }

        main_task(trainer_config)
        ray.shutdown()
