import numpy as np 
from env import CarEnv 
from torch_agent import Agent
from utils import plot_learning_curve
import sys
from tqdm import tqdm 
from settings import DISCOUNT
import os
from threading import Thread
import time
from settings import FPS, carla

if __name__ == '__main__':
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    np.random.seed(1)

    if not os.path.isdir('models'):
        os.makedirs('models')
    

    env = CarEnv()

    agent = Agent(gamma=DISCOUNT, epsilon=1.0, lr=0.0007, input_dims=[3, 480, 640], batch_size=4,\
                  n_actions=3, esp_end = 0.01, rl_algo ='DDQN', \
                  save_targ_net = 10, env_name = 'Carla0_9_9', max_mem_size=3000) #max_mem_size
    best_score = -np.inf
    resume_training = True
    inference = False

    if resume_training or inference:
        agent.load_models()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
        print("inside agent.training_initialized")

    # _ = agent.Q_local.forward(np.ones((3, env.im_height, env.im_width)))

    scores, eps_history = [], []
    n_games = 100 #2500
    # sys.exit("Exit from here")

    fname = agent.rl_algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games_' + 'mem' + str(agent.replay_mem.mem_size) + 'upd' + str(agent.repeat_save)
    figure_file = 'plots/' + fname + '.png'

    for episode in tqdm(range(n_games), unit='episodes'):
        env.collision_hist = []
        score = 0
        done  = False
        observation = env.reset()

        # print("Starting episode: ",episode)
        
        while not done:
            action = agent.choose_actions(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward 
            if not inference:
                agent.store_transition(observation, action, reward, observation_, int(done))
                # agent.learn()

            observation = observation_
        env.sensor.destroy()
        env.colsensor.destroy()
        env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-10:])
        print("agent.learn_step_counter: ",agent.learn_step_counter)

        print('episode', episode, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' %agent.epsilon)
        if episode%5 and avg_score > best_score:
            if not inference:
               agent.save_models()
            best_score = avg_score
    
    # plot_running_avg(scores, 'single_DQN500.jpg')
    plot_learning_curve(scores, eps_history, figure_file)
    print('best_score: ', best_score)

    agent.stop = True
    trainer_thread.join()
    agent.save_models()
