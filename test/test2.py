
# from envv3 import *
# from configs import ALL_PLAYER_ACTIONS

# if __name__ == '__main__':
#     # set_global_seeds(DQNSetting.SEED)
#     Game = GatherMultEnv()
#     # Game.visual = True
#     Game.reset()

#     print(Game.observation_space)
#     while True:
#         actions = {'agent-0':random.choice(ALL_PLAYER_ACTIONS), 'agent-1':random.choice(ALL_PLAYER_ACTIONS)}
#         observations, rewards, dones, info =Game.step(actions)
#         # print(rewards)
        

def compute_resource(use_gpu_for_driver,use_gpus_for_workers,num_gpus,num_cpus, num_workers_per_device):

    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    spare_gpus=None
    spare_cpus = None
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
    print(gpus_for_driver, cpus_for_driver, spare_gpus, spare_cpus, num_workers, num_gpus_per_worker, num_cpus_per_worker)