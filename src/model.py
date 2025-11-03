from transformers import AutoModel
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

    def forward(self, input_ids: torch.tensor) -> torch.tensor:
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
        return 2 * self.max_reward * rewards - self.max_reward  # Set the range into [-max, max].
