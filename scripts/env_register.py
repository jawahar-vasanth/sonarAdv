import safety_gym
import gym
import gym_mod

config = {
    'robot_base': 'xmls/dynamic_car.xml',
    'task': 'push',
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'vases_num': 4
}

test_env = gym_mod.SafexpEnvBase(name = '', config = config)
test_env.register(name='', config = config)