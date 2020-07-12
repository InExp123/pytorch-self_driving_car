import matplotlib.pyplot as plt
import numpy as np
import statistics

def plot(score_history, filename):
       
       t = [i for i in range(len(score_history))]
       fig, ax = plt.subplots()
       ax.plot(t, score_history)

       ax.set(xlabel='episodes', ylabel='reward',
              title='score per episode graph')
       ax.grid()

       fig.savefig(filename)
       plt.show()
       plt.close()

def plot_running_avg(score_history, filename):
       N = len(score_history)
       running_avg = np.empty(N)
       for t in range(N):
              running_avg[t] = statistics.mean(score_history[max(0, t-100):(t+1)])
       t = [i for i in range(N)]
       fig, ax = plt.subplots()
       ax.plot(t, running_avg)

       ax.set(xlabel='episodes', ylabel='reward',
              title="Running Average")
       ax.grid()

       fig.savefig(filename)
       plt.show()
       plt.close()

def plot_learning_curve(scores, epsilons, filename, lines=None):

       x = [i for i in range(len(scores))]
       fig=plt.figure()
       ax=fig.add_subplot(111, label="1")
       ax2=fig.add_subplot(111, label="2", frame_on=False)

       ax.plot(x, epsilons, color="C0")
       ax.set_xlabel("Training Steps", color="C0")
       ax.set_ylabel("Epsilon", color="C0")
       ax.tick_params(axis='x', colors="C0")
       ax.tick_params(axis='y', colors="C0")

       N = len(scores)
       running_avg = np.empty(N)
       for t in range(N):
              running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

       ax2.scatter(x, running_avg, color="C1")
       ax2.axes.get_xaxis().set_visible(False)
       ax2.yaxis.tick_right()
       ax2.set_ylabel('Score', color="C1")
       ax2.yaxis.set_label_position('right')
       ax2.tick_params(axis='y', colors="C1")

       if lines is not None:
              for line in lines:
                     plt.axvline(x=line)

       plt.savefig(filename)

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

# import torch 
# A = torch.randn(3, 3)
# print(A)
# b = A.gather(1, torch.tensor([0,1,1]).unsqueeze(1))
# print(b.shape)
# c = b.view(3)
# #c = b.reshape(3)
# print(c.shape)
       