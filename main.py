from copy import deepcopy
from signal import SIGINT, signal

import torch
import torch.optim as optim
from PyFlyt.pz_envs import MAQuadXHoverEnv
from wingman import ReplayBuffer, Wingman, cpuize, gpuize, shutdown_handler

from algorithms import SAC


def train(wm: Wingman):
    # grab config
    cfg = wm.cfg

    # setup env, model, replaybuffer
    env = setup_env(wm)
    model, optims = setup_nets(wm)
    memory = ReplayBuffer(cfg.buffer_size)

    wm.log["num_episodes"] = 0
    while memory.count <= cfg.total_steps:
        wm.log["num_episodes"] += 1

        """ENVIRONMENT ROLLOUT"""
        model.eval()
        model.zero_grad()

        with torch.no_grad():
            # get the initial state
            next_obs, info = env.reset()
            cumulative_reward = 0.0
            while env.agents:
                # update initial state
                obs = deepcopy(next_obs)

                # get the action from policy
                action = dict()
                for ag in env.agents:
                    action_distribution = model.actor(gpuize(obs[ag]))
                    action[ag] = cpuize(
                        model.actor.sample(*action_distribution)[0].squeeze()
                    )

                # get the next state and other stuff
                next_obs, reward, term, trunc, info = env.step(action)

                # record the reward
                cumulative_reward += sum(reward.values())
                wm.log["cumulative_reward"] = cumulative_reward

                # store stuff in mem, only append for agents with next_obs (not dead)
                for ag in next_obs:
                    memory.push(
                        [
                            obs[ag],
                            action[ag],
                            reward[ag],
                            next_obs[ag],
                            term[ag],
                        ],
                        random_rollover=cfg.random_rollover,
                    )

        """TRAINING RUN"""
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )

        for repeat_num in range(int(cfg.repeats_per_buffer)):
            for batch_num, stuff in enumerate(dataloader):
                model.train()

                obs = gpuize(stuff[0], cfg.device)
                actions = gpuize(stuff[1], cfg.device)
                rewards = gpuize(stuff[2], cfg.device)
                next_obs = gpuize(stuff[3], cfg.device)
                terms = gpuize(stuff[4], cfg.device)

                # train critic
                for _ in range(cfg.critic_update_multiplier):
                    model.zero_grad()
                    q_loss = model.calc_critic_loss(
                        obs,
                        actions,
                        rewards,
                        next_obs,
                        terms,
                    )
                    q_loss.backward()
                    optims["critic"].step()
                    model.update_q_target()

                # train actor
                for _ in range(cfg.actor_update_multiplier):
                    model.zero_grad()
                    rnf_loss = model.calc_actor_loss(obs, terms)
                    rnf_loss.backward()
                    optims["actor"].step()

                    # train entropy regularizer
                    if model.use_entropy:
                        model.zero_grad()
                        ent_loss = model.calc_alpha_loss(obs)
                        ent_loss.backward()
                        optims["alpha"].step()

                """WANDB"""
                wm.log["num_transitions"] = memory.count
                wm.log["buffer_size"] = memory.__len__()

                """WEIGHTS SAVING"""
                to_update, model_file, optim_file = wm.checkpoint(
                    loss=-float(wm.log["cumulative_reward"]),
                    step=wm.log["num_transitions"],
                )
                if to_update:
                    torch.save(model.state_dict(), model_file)

                    optim_dict = dict()
                    for key in optims:
                        optim_dict[key] = optims[key].state_dict()
                    torch.save(optim_dict, optim_file)


def display(wm: Wingman):
    cfg = wm.cfg
    env = setup_env(wm, render_mode="human")
    model, _ = setup_nets(wm)

    with torch.no_grad():
        while True:
            # get the initial state
            next_obs, info = env.reset()
            while env.agents:
                # update initial state
                obs = deepcopy(next_obs)

                # get the action from policy
                action = dict()
                for ag in env.agents:
                    action_distribution = model.actor(gpuize(obs[ag]))
                    action[ag] = cpuize(
                        model.actor.sample(*action_distribution)[0].squeeze()
                    )

                # get the next state and other stuff
                next_obs, reward, term, trunc, info = env.step(action)


def setup_env(wm: Wingman, render_mode: str | None = None):
    cfg = wm.cfg
    env = MAQuadXHoverEnv(render_mode=render_mode)
    cfg.obs_size = env.observation_space(0).shape[0]
    cfg.act_size = env.action_space(0).shape[0]

    return env


def setup_nets(wm: Wingman):
    cfg = wm.cfg

    # set up networks and optimizers
    model = SAC(
        obs_size=cfg.obs_size,
        act_size=cfg.act_size,
        entropy_tuning=cfg.use_entropy,
        target_entropy=cfg.target_entropy,
        discount_factor=cfg.discount_factor,
    ).to(cfg.device)
    actor_optim = optim.AdamW(
        model.actor.parameters(), lr=cfg.learning_rate, amsgrad=True
    )
    critic_optim = optim.AdamW(
        model.critic.parameters(), lr=cfg.learning_rate, amsgrad=True
    )
    alpha_optim = optim.AdamW([model.log_alpha], lr=0.01, amsgrad=True)

    optims = dict()
    optims["actor"] = actor_optim
    optims["critic"] = critic_optim
    optims["alpha"] = alpha_optim

    # get latest weight files
    has_weights, model_file, optim_file = wm.get_weight_files()
    if has_weights:
        # load the model
        model.load_state_dict(
            torch.load(model_file, map_location=torch.device(cfg.device))
        )

        # load the optimizer
        checkpoint = torch.load(optim_file, map_location=torch.device(cfg.device))
        for opt_key in checkpoint:
            optims[opt_key].load_state_dict(checkpoint[opt_key])

    return model, optims


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./settings.yaml")

    """ SCRIPTS HERE """

    train(wm)
    # display(wm)
