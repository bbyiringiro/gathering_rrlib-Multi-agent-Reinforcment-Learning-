import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from game_env.my_callbacks import MyCallbacks

from ray.tune import run_experiments
from ray.rllib.agents.registry import get_trainer_class


from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy

from ray.tune.registry import register_env
from model import *




from agent.basic import *
from configs import *
from game_env.multi_env import *
from utils.args_extractor  import get_args



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
    # config['env_config']['func_create'] = tune.function(env_creator)
    # config['env_config']['env_name'] = env_name
    # config['env_config']['run'] = algorithm
    # config['env_config']['N_apple'] = N
    # config['env_config']['N_tag'] = N
    # config['env_config']['N_tag'] = N

    # Calculate device configurations
    # gpus_for_driver = int(use_gpu_for_driver)
    # cpus_for_driver = 1 - gpus_for_driver
    # if use_gpus_for_workers:
    #     spare_gpus = (num_gpus - int(gpus_for_driver))
    #     num_workers = int(spare_gpus * num_workers_per_device)
    #     num_gpus_per_worker = spare_gpus / num_workers
    #     num_cpus_per_worker = 0
    # else:
    #     print(num_cpus)
    #     spare_cpus = (int(num_cpus) - int(cpus_for_driver))
    #     num_workers = int(spare_cpus * num_workers_per_device)
    #     num_gpus_per_worker = 0
    #     num_cpus_per_worker = spare_cpus / num_workers



    # hyperparams
    config.update({
        "env": env_name,
        "env_config": {
            "num_agents": num_agents,
            "visual":False,
            "exp":'optmizing',
            "n_tag":args.n_tag,
            "n_apple":args.n_apple,
            "init":False,
            "imrl":{'use':False},
            "env_name":env_name,
            "run":algorithm,
            "func_create":tune.function(env_creator),
        },
        "callbacks": MyCallbacks,

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(policy_mapping_fn)
        },
        "model": {
            'fcnet_hiddens':[32,32],
        },
        "framework": 'tf',

        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        # "rollout_fragment_length": 50,
        "horizon": 1000,
        #     # # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "dueling": False,
        "double_q": True,
            # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1, #0.02
            "epsilon_timesteps": int(2e6),  # Timesteps over which to anneal epsilon. # 500000
        },
        "lr": 0.00025, #5e-4,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,



        # # The number of contiguous environment steps to replay at once. This may
        # # be set to greater than 1 to support recurrent models.
        # "replay_sequence_length": 1,
        "lr_schedule": None,
        # "lr_schedule":
        # [[0, hparams['lr_init']],
        #     [20000000, hparams['lr_final']]],
        # "entropy_coeff": hparams['entropy_coeff'],
        "num_envs_per_worker": 8,
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_workers":1, # num_workers,
        # "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": 1,#cpus_for_driver,
        # "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
        "num_cpus_per_worker": 1,#num_cpus_per_worker,   # Can be a fraction
        # General
        # "learning_starts": 1000,
        # "train_batch_size": int(train_batch_size),
        # "buffer_size": int(1e5),
        # "compress_observations": True,
        # "rollout_fragment_length": 16,
        # "gamma": .99,
        # # "n_step": 3,
        # "prioritized_replay_alpha": 0.5,
        # "final_prioritized_replay_beta": 1.0,
        # "target_network_update_freq": 500,


    })

    config['env_config'].update({
        'imrl':{"use":True,
                "imrl_reward_alpha":1,
                "full_obs":False,
                "fairness_gamma":0.99,
                "fairness_alpha": 1,
                "fairness_epsilon":0.1,
                "reward_gamma": 0.99,
                "reward_alpha": 1,
                "aspirational": 0.5,
                "aspiration_beta": 0.1,
                "f_u": 1,
                "g_v": 1,
                "core":'fw', #wf
                "wellbeing_fx":'variance'
        }
    })

    # print(f"works {num_workers}, gpu for drive {gpus_for_driver}, cpu for drive {cpus_for_driver}, num_works on gpu {num_gpus_per_worker} , workers cpu {num_cpus_per_worker}")








    return algorithm, env_name, config


def main(args):
    # ray.init(address="auto")
    ray.init()
    hparams = gathering_params

    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)
    learning_type = 'imrl' if args.imrl else ''
    affix = f"T{args.n_tag}_P{args.n_apple}_{learning_type}"
    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm+affix
    else:
        exp_name = args.exp_name+affix

    print('starting experiment', exp_name)
    print(args.resume)
    tune.run(alg_run,
             name=exp_name,
             stop= {
                "training_iteration": args.training_iterations
            },
            checkpoint_freq = args.checkpoint_frequency,
            config = config,
            checkpoint_at_end = True,
            verbose=args.verbose,
            resume=args.resume,
            reuse_actors=args.reuse_actors
        )


if __name__ == '__main__':
    args, device = get_args()
    main(args)