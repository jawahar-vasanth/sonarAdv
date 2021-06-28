#!/usr/bin/env python
import gym
import sonarCarV0
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork



def main(algo, seed, exp_name, cpu):

    # Verify experiment
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    assert algo in algo_list, "Invalid algo"

    # Hyperparameters
    exp_name = algo
    num_steps = 1e6
    steps_per_epoch = 10000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed,data_dir='/home/jawaharvasanth/Desktop/DDP/sonarModif/data')

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    algo(env_fn=lambda: gym.make('sonarCar-v0',render_mode=True ,sensor_display= True),
         ac_kwargs=dict(
             hidden_sizes=(128, 128),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.algo, args.seed, exp_name, args.cpu)