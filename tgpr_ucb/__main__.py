import argparse
import yaml

from tgpr_ucb.verl_trainer import TGPRVerlTrainer
from tgpr_ucb.inference import DS1000Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="data/ds1000_test.parquet")
    parser.add_argument("--output", type=str, default="results/ds1000_eval.jsonl")
    parser.add_argument("--lora_path", type=str, default="")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        trainer = TGPRVerlTrainer(args.config)
        trainer.train()
    else:
        use_lora = config["model"].get("use_lora", False)
        evaluator = DS1000Evaluator(
            args.model_path or config["model"]["name"],
            config,
            use_lora=use_lora and args.lora_path,
            lora_path=args.lora_path
        )
        evaluator.evaluate(args.dataset, args.output)


if __name__ == "__main__":
    main()
