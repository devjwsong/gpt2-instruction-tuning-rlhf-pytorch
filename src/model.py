from typing import Tuple

from transformers import AutoModel, AutoModelForCausalLM
from torch import nn
import torch


class RewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.reward_layer = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.reward_layer.weight)

    def forward(self, last_hidden_states: torch.tensor) -> torch.tensor:
        # last_hidden_states: (B, d)
        return self.reward_layer(last_hidden_states).squeeze(-1)  # (B)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, init_bias: float=3.0):
        super().__init__()
        self.value_layer = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.value_layer.weight)
        self.value_layer.bias.data.fill_(init_bias)

    def forward(self, last_hidden_states: torch.tensor) -> torch.tensor:
        # last_hidden_states: (B, L, d)
        return self.value_layer(last_hidden_states).squeeze(-1)  # (B, L)


class RewardModel(nn.Module):
    def __init__(self, sf_model_path: str, reward_token_id: int, max_reward: float=1.0):
        super().__init__()

        self.model = AutoModel.from_pretrained(sf_model_path)
        self.reward_head = RewardHead(self.model.config.n_embd)
        self.reward_token_id = reward_token_id
        self.max_reward = max_reward

    def _find_reward_tokens(self, input_ids: torch._tensor) -> torch.tensor:
        # input_ids: (B, L)
        masks = (input_ids == self.reward_token_id).int()  # (B, L)
        cumsums = torch.cumsum(masks, dim=1)  # (B, L)
        locs = (cumsums == 1) & masks  # (B, L)
        return torch.nonzero(locs, as_tuple=True)[1]  # (B)

    def forward(self, input_ids: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # input_ids: (B, L)
        # Check the location of the reward token.
        reward_locs = self._find_reward_tokens(input_ids)  # (B)

        # Get the last hidden state vectors.
        outputs = self.model(input_ids, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state  # (B, L, d)

        # Apply the reward head only on the reward token location.
        batch_size, _, hidden_size = last_hidden_states.shape
        gather_locs = reward_locs.view(batch_size, 1, 1).expand(-1, -1, hidden_size)  # (B, L, d)
        hidden_states = torch.gather(last_hidden_states, dim=1, index=gather_locs).squeeze(1)  # (B, d)
        rewards = self.reward_head(hidden_states)  # (B)

        # Reward normalization.
        rewards = torch.sigmoid(rewards)  # (B)
        return 1.0 + (self.max_reward - 1) * rewards, reward_locs  # Set the range into [1.0 - max_reward].
    

class PolicyWithValueHead(nn.Module):
    def __init__(self, sf_model_path: str, init_bias: float=3.0):
        super().__init__()

        # The policy is a causal LM same as SFT model.
        self.policy = AutoModelForCausalLM.from_pretrained(sf_model_path)
        self.value_head = ValueHead(self.policy.config.n_embd, init_bias)

    def forward(self, input_ids: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # input_ids: (B, L)

        # Get the last hidden state vectors.
        outputs = self.policy(input_ids, output_hidden_states=True)
        lm_logits = outputs.logits  # (B, L, V)
        last_hidden_states = outputs.hidden_states[-1]  # (B, L, d)

        # Apply the value head for all tokens.
        values = self.value_head(last_hidden_states)  # (B, L)
        return lm_logits, values
    
    def generate(self, *args, **kwargs):
        return self.policy.generate(*args, **kwargs)

