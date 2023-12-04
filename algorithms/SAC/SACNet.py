import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from wingman import NeuralBlocks


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [act_size + obs_size, 256, 256, 1]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states, actions):
        if len(actions.shape) != len(states.shape):
            states = torch.stack([states] * actions.shape[0], dim=0)

        output = torch.cat((states, actions), dim=-1)
        output = self.net(output)

        return output


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles
    """

    def __init__(self, act_size, obs_size, num_networks=2):
        super().__init__()

        networks = [Critic(act_size, obs_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states, actions):
        """
        states is of shape B x input_shape
        actions is of shape B x act_size
        output is a tuple of B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(states, actions))

        output = torch.cat(output, dim=-1)

        return output


class GaussianActor(nn.Module):
    """
    Gaussian Actor
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [obs_size, 256, 256, act_size * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.net(states).reshape(-1, 2, self.act_size).permute(1, 0, 2)
        return output[0], output[1]

    @staticmethod
    def sample(mu, sigma):
        """
        output:
            actions is of shape B x act_size
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma + 1) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @staticmethod
    def infer(mu, sigma):
        return torch.tanh(mu)
