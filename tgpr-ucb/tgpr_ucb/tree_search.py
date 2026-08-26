import math
import torch


class TreeNode:
    def __init__(self, code, depth=0):
        self.code = code
        self.depth = depth
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.reward_sq = 0.0
        self.mean_reward = 0.0
        self.var_reward = 0.0
        self.parent = None


class UCBTreeSearch:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.max_nodes = config["tree"]["max_nodes"]
        self.c = config["tree"]["exploration_const"]
        self.max_depth = config["tree"]["max_depth"]
        self.max_gen_len = config["training"]["max_gen_len"]
        self.temp = config["rollout"]["temperature"]
        self.top_p = config["rollout"]["top_p"]

    def search(self, prompt, group_size, reward_fn):
        root = TreeNode(prompt, depth=0)
        nodes = [root]

        for _ in range(self.max_nodes):
            if not nodes:
                break

            leaf = self._select(nodes)
            if leaf.depth >= self.max_depth:
                continue

            children = self._expand(leaf)
            for child in children:
                child.parent = leaf
                reward = self._eval(child, reward_fn)
                child.visits = 1
                child.total_reward = reward
                child.reward_sq = reward ** 2
                child.mean_reward = reward
                nodes.append(child)
                leaf.children.append(child)

            avg_reward = sum(c.mean_reward for c in children) / max(len(children), 1)
            self._backprop(leaf, avg_reward)

        candidates = self._collect(root, group_size)
        codes = [c.code for c in candidates]
        rewards = [c.mean_reward for c in candidates]
        return codes, rewards

    def _select(self, nodes):
        best_score = -float('inf')
        best_node = nodes[0]
        total_visits = sum(n.visits for n in nodes)

        for node in nodes:
            if node.visits == 0:
                return node

            exploitation = node.mean_reward
            exploration = self.c * math.sqrt(
                math.log(total_visits + 1) / node.visits
            ) * min(
                0.25,
                math.sqrt(
                    (
                        node.var_reward
                        + math.sqrt(2 * math.log(total_visits + 1) / node.visits)
                    ) / node.visits
                )
            )

            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _expand(self, node):
        children = []
        for _ in range(3):
            continuation = self._generate(node.code)
            children.append(TreeNode(continuation, depth=node.depth + 1))
        return children

    def _generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_gen_len,
                temperature=self.temp,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return prompt + generated

    def _eval(self, node, reward_fn):
        return reward_fn.compute(node.code, [node.code])

    def _backprop(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node.reward_sq += reward ** 2
            node.mean_reward = node.total_reward / node.visits
            if node.visits > 1:
                node.var_reward = max(
                    0,
                    (node.reward_sq / node.visits) - (node.mean_reward ** 2)
                )
            node = node.parent

    def _collect(self, root, k):
        all_nodes = []
        self._dfs(root, all_nodes)
        all_nodes.sort(key=lambda x: x.mean_reward, reverse=True)
        return all_nodes[:k]

    def _dfs(self, node, acc):
        if node.visits > 0:
            acc.append(node)
        for child in node.children:
            self._dfs(child, acc)
