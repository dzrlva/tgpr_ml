import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import numpy as np

from tgpr_ucb.tree_search import UCBTreeSearch
from tgpr_ucb.reward import RewardFunction


class GRPOTrainer:
    def __init__(self, config, dataset, corpus, output_dir):
        self.config = config
        self.dataset = dataset
        self.corpus = corpus
        self.output_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"]
        )

        self.tree_search = UCBTreeSearch(config, self.model, self.tokenizer)
        self.reward_fn = RewardFunction(config, corpus)

        self.kl_coef = config["training"]["kl_coef"]
        self.group_size = config["training"]["group_size"]
        self.grad_accum = config["training"]["grad_accum"]

    def train(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=self.dataset.collate
        )

        self.model.train()
        step = 0

        for epoch in range(self.config["training"]["epochs"]):
            for batch_idx, batch in enumerate(loader):
                prompts = batch["prompts"]

                all_completions = []
                all_rewards = []
                all_advantages = []

                for prompt in prompts:
                    completions, rewards = self.tree_search.search(
                        prompt, self.group_size, self.reward_fn
                    )
                    mean_r = np.mean(rewards)
                    std_r = np.std(rewards) + 1e-6
                    advantages = [(r - mean_r) / std_r for r in rewards]

                    all_completions.extend(completions)
                    all_rewards.extend(rewards)
                    all_advantages.extend(advantages)

                loss = self.grpo_step(
                    prompts * self.group_size,
                    all_completions,
                    all_advantages
                )

                if (batch_idx + 1) % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1

                    if step % 100 == 0:
                        print(
                            f"step {step} | "
                            f"reward: {np.mean(all_rewards):.3f} | "
                            f"loss: {loss.item():.3f}"
                        )
                    if step % 500 == 0:
                        self.save_checkpoint(step)

    def grpo_step(self, prompts, completions, advantages):
        total_loss = 0.0

        for prompt, completion, advantage in zip(prompts, completions, advantages):
            text = prompt + completion
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self.config["training"]["max_prompt_len"]
                    + self.config["training"]["max_gen_len"]
                )
            ).to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(**inputs, labels=inputs.input_ids)
                log_probs = -outputs.loss
                loss = -advantage * log_probs
                total_loss += loss

        total_loss = total_loss / len(prompts)
        total_loss.backward()
        return total_loss

    def save_checkpoint(self, step):
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"saved checkpoint to {path}")
