#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import matplotlib.pyplot as plt


def run_policy(env, get_action, max_ep_len=None, num_episodes=5, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    ep_retr = []
    ep_lenr = []
    ep_collis = 0
    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            # time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        collis =  info.get('crash_info')
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            ep_retr.append(ep_ret)
            ep_lenr.append(ep_len)
            # print(collis)
            if collis: ep_collis += 1
            # print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    print('Time taken %d \t EpRet_Avg %.3f'%(time.time()-start_time,np.average(ep_retr)))
    print('EpLen %d \t Avg_Collisions %.3f'%(np.average(ep_lenr),ep_collis/num_episodes ))
    # logger.log_tabular('EpRet', with_min_and_max=True)
    # logger.log_tabular('EpCost', with_min_and_max=True)
    # logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular()

    return np.average(ep_retr), np.average(ep_lenr),ep_collis/num_episodes 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=2000)
    parser.add_argument('--episodes', '-n', type=int, default=25)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    ep_rr = []
    ep_lr = []
    ep_col = []
    rang = [10*i for i in range(0,int(args.itr/10))]
    for i in rang:
        env, get_action, sess = load_policy(args.fpath,
                                            i if i >=0 else 'last',args.deterministic)
        print("Itr:",i)
        ep_r, ep_l, coll = run_policy(env, get_action, args.len, args.episodes, not(args.norender))
        ep_rr.append(ep_r)
        ep_lr.append(ep_l)
        ep_col.append(coll)
    env, get_action, sess = load_policy(args.fpath,'last', args.deterministic)
    ep_r, ep_l, coll = run_policy(env, get_action, args.len, args.episodes, not(args.norender))
    ep_rr.append(ep_r)
    ep_lr.append(ep_l)
    ep_col.append(coll)
    rang.append(args.itr)
    file_name = args.fpath + "test_policy.txt"
    with open(file_name, 'w') as file:
        file.write('Itr'+'\t'+'EpRet'+'\t'+'EpLen'+'\t'+'EpColl'+'\n')
        for i in range(len(rang)):
            s = str(rang[i])
            s = s +'\t' + str(ep_rr[i])
            s = s +'\t' + str(ep_lr[i])
            s = s +'\t' + str(ep_col[i])
            file.write(s+'\n')

    plt.subplot(3,1,1)
    plt.plot(rang,ep_rr)
    plt.title("Average Episode Return")
    plt.subplot(3,1,2)
    plt.plot(rang,ep_lr)
    plt.title("Average Episode Length") 
    plt.subplot(3,1,3)
    plt.plot(rang,ep_col)
    plt.title("Average Episode Collision")
    plt.show()

        
