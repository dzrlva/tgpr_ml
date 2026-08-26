import os
import json
import subprocess
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from tgpr_ucb.data.ds1000 import DS1000Dataset


class DS1000Evaluator:
    def __init__(self, model_path, config, use_lora=False, lora_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        if use_lora and lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.max_gen_len = config["training"]["max_gen_len"]
        self.temp = config["rollout"]["temperature"]
        self.top_p = config["rollout"]["top_p"]

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_gen_len,
                temperature=self.temp,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    def evaluate(self, dataset_path, output_path="results/ds1000_results.jsonl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset = DS1000Dataset(dataset_path)
        results = []
        correct = 0
        total = 0

        for item in tqdm(dataset, desc="evaluating"):
            prompt = item["prompt"]
            generated = self.generate(prompt)

            test_code = item.get("test_code", "")
            code_context = item.get("code_context", "")

            passed = self._run_test(generated, code_context, test_code)

            results.append({
                "prompt": prompt,
                "generated": generated,
                "passed": passed,
                "lib": item.get("lib", "unknown")
            })

            if passed:
                correct += 1
            total += 1

        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        accuracy = correct / total if total > 0 else 0
        print(f"DS-1000 accuracy: {accuracy:.3f} ({correct}/{total})")

        by_lib = {}
        for r in results:
            lib = r["lib"]
            if lib not in by_lib:
                by_lib[lib] = {"correct": 0, "total": 0}
            by_lib[lib]["total"] += 1
            if r["passed"]:
                by_lib[lib]["correct"] += 1

        for lib, stats in by_lib.items():
            acc = stats["correct"] / stats["total"]
            print(f"  {lib}: {acc:.3f} ({stats['correct']}/{stats['total']})")

        return accuracy, results

    def _run_test(self, generated, code_context, test_code):
        full_code = code_context + "\n" + generated + "\n" + test_code

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            path = f.name

        try:
            result = subprocess.run(
                ["python", path],
                capture_output=True,
                text=True,
                timeout=10
            )
            os.unlink(path)
            return result.returncode == 0
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            return False
