import os
import yaml
import argparse

from tgpr_ucb.grpo_trainer import GRPOTrainer
from tgpr_ucb.data.dataset import CodeDataset
from tgpr_ucb.data.corpus import ValidatedCorpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CodeDataset(args.data_path, config["training"]["max_prompt_len"])
    corpus = ValidatedCorpus(config["reward"]["corpus_path"])

    trainer = GRPOTrainer(config, dataset, corpus, args.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
