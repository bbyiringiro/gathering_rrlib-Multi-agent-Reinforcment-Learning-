# # #     # === Exploration Settings ===
#     config.update({
# # #     # === Model ===
# # #     # Number of atoms for representing the distribution of return. When
# # #     # this is greater than 1, distributional Q-learning is used.
# # #     # the discrete supports are bounded by v_min and v_max
# # #     "num_atoms": 1,
# # #     "v_min": -10.0,
# # #     "v_max": 10.0,
# # #     # Whether to use noisy network
# # #     "noisy": False,
# # #     # control the initial value of noisy nets
# # #     "sigma0": 0.5,
# # #     # Whether to use dueling dqn
#     "dueling": False,
# # #     # Dense-layer setup for each the advantage branch and the value branch
# # #     # in a dueling architecture.
# # #     # "hiddens": [256],
# # #     # Whether to use double dqn
#     "double_q": False,
# # #     # N-step Q learning
# # #     "n_step": 1,

# # #     # === Exploration Settings ===
#     "exploration_config": {
#         # The Exploration class to use.
#         "type": "EpsilonGreedy",
#         # Config for the Exploration class' constructor:
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.1, #0.02
#         "epsilon_timesteps": 500000,  # Timesteps over which to anneal epsilon. # 500000

#         # For soft_q, use:
#         # "exploration_config" = {
#         #   "type": "SoftQ"
#         #   "temperature": [float, e.g. 1.0]
#         # }
#     },
#     "evaluation_interval":5,
#     "evaluation_num_episodes":10,
#     "evaluation_num_workers":1,
#     "evaluation_config": {
#         "explore": True,
#         "exploration_config": {
#         # The Exploration class to use.
#         "type": "EpsilonGreedy",
#         # Config for the Exploration class' constructor:
#         "initial_epsilon": 0.1,
#         "final_epsilon": 0.1, #0.02
#         "epsilon_timesteps": 1,  # Timesteps over which to anneal epsilon. # 500000

#         # For soft_q, use:
#         # "exploration_config" = {
#         #   "type": "SoftQ"
#         #   "temperature": [float, e.g. 1.0]
#         # }
#         },

#     },

# # #     # # Minimum env steps to optimize for per train call. This value does
# # #     # # not affect learning, only the length of iterations.
# #     "timesteps_per_iteration": 1000,
# # #     # # Update the target network every `target_network_update_freq` steps.
# # #     # "target_network_update_freq": 500,
# # #     # # === Replay buffer ===
# # #     # # Size of the replay buffer. Note that if async_updates is set, then
# # #     # # each worker will have a replay buffer of this size.
# #     "buffer_size": 50000,
# # #     # # The number of contiguous environment steps to replay at once. This may
# # #     # # be set to greater than 1 to support recurrent models.
# # #     # "replay_sequence_length": 1,
# # #     # # If True prioritized replay buffer will be used.
#     "prioritized_replay": False,#True,
# # #     # # Alpha parameter for prioritized replay buffer.
# # #     # "prioritized_replay_alpha": 0.6,
# # #     # # Beta parameter for sampling from prioritized replay buffer.
# # #     # "prioritized_replay_beta": 0.4,
# # #     # # Final value of beta (by default, we use constant beta=0.4).
# # #     # "final_prioritized_replay_beta": 0.4,
# # #     # # Time steps over which the beta parameter is annealed.
# # #     # "prioritized_replay_beta_annealing_timesteps": 20000,
# # #     # # Epsilon to add to the TD errors when updating priorities.
# # #     # "prioritized_replay_eps": 1e-6,

# # #     # # Whether to LZ4 compress observations
# # #     # "compress_observations": False,
# # #     # # Callback to run before learning on a multi-agent batch of experiences.
# # #     # "before_learn_on_batch": None,
# # #     # # If set, this will fix the ratio of replayed from a buffer and learned on
# # #     # # timesteps to sampled from an environment and stored in the replay buffer
# # #     # # timesteps. Otherwise, the replay will proceed at the native ratio
# # #     # # determined by (train_batch_size / rollout_fragment_length).
# # #     # "training_intensity": None,

# # #     # === Optimization ===
# # #     # Learning rate for adam optimizer
#     "lr": 0.00025, #5e-4,
# # #     # Learning rate schedule
# # #     "lr_schedule": None,
# # #     # Adam epsilon hyper parameter
# #     "adam_epsilon": 1e-8, # one used in the other env0.01
# # #     # If not None, clip gradients during optimization at this value
# # #     "grad_clip": 40,
# # #     # How many steps of the model to sample before learning starts.
# #     # "learning_starts": 100,



# # #     # === Parallelism ===
# # #     # Number of workers for collecting samples with. This only makes sense
# # #     # to increase if your environment is particularly slow to sample, or if
# # #     # you"re using the Async or Ape-X optimizers.
# # #     "num_workers": 0,
# # #     # Whether to compute priorities on workers.
# # #     "worker_side_prioritization": False,
# # #     # Prevent iterations from going lower than this time span
# # #     "min_iter_time_s": 1,
#     })    
    
    # config.update({
    #     # For running in editor, force to use just one Worker (we only have
    #     # one Unity running)!
    #     "num_workers": 0,
    #     # Other settings.
    #     "lr": 0.0003,
    #     # "lambda": 0.95,
        # "gamma": 0.99,
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







#     #ICM
#     # Intrinsic Curiosity

#     # config.update({
#     #     'exploration_config':{
#     #         "type": "Curiosity",  # <- Use the Curiosity module for exploring.
#     #         "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
#     #         "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
#     #         "feature_dim": 288,  # Dimensionality of the generated feature vectors.
#     #         # Setup of the feature net (used to encode observations into feature (latent) vectors).
#     #         "feature_net_config": {
#     #             "fcnet_hiddens": [],
#     #             "fcnet_activation": "relu",
#     #         },
#     #         "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
#     #         "inverse_net_activation": "relu",  # Activation of the "inverse" model.
#     #         "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
#     #         "forward_net_activation": "relu",  # Activation of the "forward" model.
#     #         "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
#     #         # Specify, which exploration sub-type to use (usually, the algo's "default"
#     #         # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
#     #         "sub_exploration": {"type": "StochasticSampling",}
#     #     }
#     # })

#     # config["exploration_config"] = {
#     #         "type": "Curiosity",
#     #         "eta": 0.1,
#     #         "lr": 0.001,
#     #         # No actual feature net: map directly from observations to feature
#     #         # vector (linearly).
#     #         "feature_net_config": {
#     #             "fcnet_hiddens": [],
#     #             "fcnet_activation": "relu",
#     #         },
#     #         "sub_exploration": {
#     #             "type": "StochasticSampling",
#     #         },
#     #         "forward_net_activation": "relu",
#     #         "inverse_net_activation": "relu",
#     #     }