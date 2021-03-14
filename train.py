import argparse

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from ray.tune import run_experiments
from ray.rllib.agents.registry import get_trainer_class


from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy

from ray.tune.registry import register_env
# from model import *




from agent.basic import *
from configs import *
from game_env.multi_env import *
from utils.args_extractor  import get_args


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # print("episode {} (env-idx={}) started.".format(
        #     episode.episode_id, env_index))

        episode.user_data["extrinsic_reward"] = []
        episode.user_data["intrinsic_reward"] = []
        episode.user_data["aggresseviness"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        
        in_reward = 0
        ex_reward = 0
        tag_action = 0
        for i in range(2):
            agent_key = 'agent-'+str(i)
            
            info = episode.last_info_for(agent_key)
            if not info: return 
            
            ex_reward +=info.get('exR',0)
            in_reward += info.get('inR',0)
            if info['agent_action'] == PlayerAction.USE_BEAM:
                tag_action += 1
        episode.user_data["extrinsic_reward"].append(ex_reward)
        episode.user_data["intrinsic_reward"].append(in_reward)
        episode.user_data["aggresseviness"].append(tag_action)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        episode.custom_metrics["ExReward"] = np.sum(episode.user_data["extrinsic_reward"])
        episode.custom_metrics["InReward"] = np.sum(episode.user_data["intrinsic_reward"])
        episode.custom_metrics["aggresseviness"] = np.sum(episode.user_data["aggresseviness"])
        
        

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
    #                       result: dict, **kwargs) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #         policy, result["sum_actions_in_train_batch"]))

    # def on_postprocess_trajectory(
    #         self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: str, policy_id: str, policies: Dict[str, Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict[str, SampleBatch], **kwargs):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1

gathering_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    # 'entropy_coeff': .000687
    }



def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    def env_creator(env_config):
        return GatherMultEnv(env_config)
    single_env = GatherMultEnv({'visual':False, 'init':True})
    

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        config={}
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policies = {}
    for i in range(num_agents):
        policies['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    # model_name = "conv_to_fc_net"
    # ModelCatalog.register_custom_model(model_name, VisionNetwork2)

    agent_cls = get_trainer_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm
    # config['env_config']['N_apple'] = N
    # config['env_config']['N_tag'] = N
    # config['env_config']['N_tag'] = N

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - int(gpus_for_driver))
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        print(num_cpus)
        spare_cpus = (int(num_cpus) - int(cpus_for_driver))
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers



    # hyperparams
    config.update({
        "env": env_name,
        "env_config": {
            "num_agents": num_agents,
            "visual":False,
            "n_tag":args.n_tag,
            "n_apple":args.n_apple,
            "init":False,
            "imrl":args.imrl,
            "full_obs":args.full_obs,
        },
        "callbacks": MyCallbacks,
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(policy_mapping_fn)
        },
        "model": {
            # "custom_model": model_name,
            'fcnet_hiddens':[32,32],
            # "use_attention": True,
            # "use_lstm": True,
            # "lstm_cell_size": 50

            # "lstm_use_prev_action":True,
            # "lstm_use_prev_reward":True,
        
        },
        "framework": args.framework,
        "train_batch_size": train_batch_size,
        "horizon": 1000,
        "rollout_fragment_length":50,
        # "lr_schedule":
        # [[0, hparams['lr_init']],
        #     [20000000, hparams['lr_final']]],
        # "entropy_coeff": hparams['entropy_coeff'],
        "num_workers": num_workers,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
    })

    print(num_workers, gpus_for_driver, cpus_for_driver, num_gpus_per_worker, num_cpus_per_worker)

#     # === Exploration Settings ===
    config.update({
#     # === Model ===
#     # Number of atoms for representing the distribution of return. When
#     # this is greater than 1, distributional Q-learning is used.
#     # the discrete supports are bounded by v_min and v_max
#     "num_atoms": 1,
#     "v_min": -10.0,
#     "v_max": 10.0,
#     # Whether to use noisy network
#     "noisy": False,
#     # control the initial value of noisy nets
#     "sigma0": 0.5,
#     # Whether to use dueling dqn
#     "dueling": False,
#     # Dense-layer setup for each the advantage branch and the value branch
#     # in a dueling architecture.
#     # "hiddens": [256],
#     # Whether to use double dqn
#     "double_q": False,
#     # N-step Q learning
#     "n_step": 1,

#     # === Exploration Settings ===
    "exploration_config": {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05, #0.02
        "epsilon_timesteps": 500000,  # Timesteps over which to anneal epsilon. # 500000

        # For soft_q, use:
        # "exploration_config" = {
        #   "type": "SoftQ"
        #   "temperature": [float, e.g. 1.0]
        # }
    },
#     # Switch to greedy actions in evaluation workers.
#     "evaluation_config": {
#         "explore": False,
#     },

#     # # Minimum env steps to optimize for per train call. This value does
#     # # not affect learning, only the length of iterations.
#     # "timesteps_per_iteration": 1000,
#     # # Update the target network every `target_network_update_freq` steps.
#     # "target_network_update_freq": 500,
#     # # === Replay buffer ===
#     # # Size of the replay buffer. Note that if async_updates is set, then
#     # # each worker will have a replay buffer of this size.
#     # "buffer_size": 50000,
#     # # The number of contiguous environment steps to replay at once. This may
#     # # be set to greater than 1 to support recurrent models.
#     # "replay_sequence_length": 1,
#     # # If True prioritized replay buffer will be used.
#     # "prioritized_replay": True,
#     # # Alpha parameter for prioritized replay buffer.
#     # "prioritized_replay_alpha": 0.6,
#     # # Beta parameter for sampling from prioritized replay buffer.
#     # "prioritized_replay_beta": 0.4,
#     # # Final value of beta (by default, we use constant beta=0.4).
#     # "final_prioritized_replay_beta": 0.4,
#     # # Time steps over which the beta parameter is annealed.
#     # "prioritized_replay_beta_annealing_timesteps": 20000,
#     # # Epsilon to add to the TD errors when updating priorities.
#     # "prioritized_replay_eps": 1e-6,

#     # # Whether to LZ4 compress observations
#     # "compress_observations": False,
#     # # Callback to run before learning on a multi-agent batch of experiences.
#     # "before_learn_on_batch": None,
#     # # If set, this will fix the ratio of replayed from a buffer and learned on
#     # # timesteps to sampled from an environment and stored in the replay buffer
#     # # timesteps. Otherwise, the replay will proceed at the native ratio
#     # # determined by (train_batch_size / rollout_fragment_length).
#     # "training_intensity": None,

#     # === Optimization ===
#     # Learning rate for adam optimizer
#     "lr": 0.00025, #5e-4,
#     # Learning rate schedule
#     "lr_schedule": None,
#     # Adam epsilon hyper parameter
#     "adam_epsilon": 1e-8,
#     # If not None, clip gradients during optimization at this value
#     "grad_clip": 40,
#     # How many steps of the model to sample before learning starts.
#     "learning_starts": 1000,
#     # Update the replay buffer with this many samples at once. Note that
#     # this setting applies per-worker if num_workers > 1.
#     "rollout_fragment_length": 4,
#     # Size of a batch sampled from replay buffer for training. Note that
#     # if async_updates is set, then each worker returns gradients for a
#     # batch of this size.
#     "train_batch_size": 32,

#     # === Parallelism ===
#     # Number of workers for collecting samples with. This only makes sense
#     # to increase if your environment is particularly slow to sample, or if
#     # you"re using the Async or Ape-X optimizers.
#     "num_workers": 0,
#     # Whether to compute priorities on workers.
#     "worker_side_prioritization": False,
#     # Prevent iterations from going lower than this time span
#     "min_iter_time_s": 1,
    })

    # config.update({

    #     # For running in editor, force to use just one Worker (we only have
    #     # one Unity running)!
    #     "num_workers": 0,
    #     # Other settings.
    #     "lr": 0.0003,
    #     # "lambda": 0.95,
    #     "gamma": 0.99,
    #     # "sgd_minibatch_size": 256,
    #     "train_batch_size": 4000,
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #     # "num_sgd_iter": 20,
    #     "rollout_fragment_length": 200,
    #     # "clip_param": 0.2,
    #     # Multi-agent setup for the particular env.
    #     "model": {
    #         "fcnet_hiddens": [512, 512],
    #     },
    #     "no_done_at_end": True,
    # })
    # Switch on Curiosity based exploration for Pyramids env
    # (not solvable otherwise).
    # config["exploration_config"] = {
    #     "type": "Curiosity",
    #     "eta": 0.1,
    #     "lr": 0.001,
    #     # No actual feature net: map directly from observations to feature
    #     # vector (linearly).
    #     "feature_net_config": {
    #         "fcnet_hiddens": [],
    #         "fcnet_activation": "relu",
    #     },
    #     "sub_exploration": {
    #         "type": "StochasticSampling",
    #     },
    #     "forward_net_activation": "relu",
    #     "inverse_net_activation": "relu",
    # }

    #ICM
    # Intrinsic Curiosity

    # config.update({
    #     'exploration_config':{
    #         "type": "Curiosity",  # <- Use the Curiosity module for exploring.
    #         "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
    #         "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
    #         "feature_dim": 288,  # Dimensionality of the generated feature vectors.
    #         # Setup of the feature net (used to encode observations into feature (latent) vectors).
    #         "feature_net_config": {
    #             "fcnet_hiddens": [],
    #             "fcnet_activation": "relu",
    #         },
    #         "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
    #         "inverse_net_activation": "relu",  # Activation of the "inverse" model.
    #         "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
    #         "forward_net_activation": "relu",  # Activation of the "forward" model.
    #         "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    #         # Specify, which exploration sub-type to use (usually, the algo's "default"
    #         # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    #         "sub_exploration": {"type": "StochasticSampling",}
    #     }
    # })

    # config["exploration_config"] = {
    #         "type": "Curiosity",
    #         "eta": 0.1,
    #         "lr": 0.001,
    #         # No actual feature net: map directly from observations to feature
    #         # vector (linearly).
    #         "feature_net_config": {
    #             "fcnet_hiddens": [],
    #             "fcnet_activation": "relu",
    #         },
    #         "sub_exploration": {
    #             "type": "StochasticSampling",
    #         },
    #         "forward_net_activation": "relu",
    #         "inverse_net_activation": "relu",
    #     }



    return algorithm, env_name, config


def main(args):
    ray.init()
    hparams = gathering_params
 
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    affix = f"T{args.n_tag}_ P{args.n_apple}"
    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm+affix
    else:
        exp_name = args.exp_name+affix
    print('Commencing experiment', exp_name)
    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
        }
    }, verbose=args.verbose, resume=args.resume, reuse_actors=args.reuse_actors)


if __name__ == '__main__':
    args, device = get_args()
    main(args)
