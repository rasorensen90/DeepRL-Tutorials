import matplotlib.pyplot as plt
import csv
import statistics as stat
import numpy as np

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams["font.family"] = "Times New Roman"

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
figuresize = (8,12) #inches

lw=1.0
ms=12
width = 0.5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure',figsize=figuresize)

network = 'DQN'

if (network == 'DQN'):
    times = ['2020-08-13-13', '2020-08-16-21']
elif (network == 'GAT'):
    times = ['', '']
elif (network == 'GGNN'):
    times = ['', '']
elif (network == 'NN'):
    times = ['', '']
elif (network == 'SAGE'):
    times = ['', '']
elif (network == 'GCN'):
    times = ['', '']
elif (network == 'CG'):
    times = ['', '']
elif (network == 'SGN'):
    times = ['', '']
elif (network == 'GIN'):
    times = ['', '']
elif (network == 'PNA'):
    times = ['', '']

env2_RF = 'Results/' + network + '/' + network + '_' + times[0] + '.csv'
env2_RT = 'Results/' + network + '/' + network + '_' + times[1] + '.csv'

fig = plt.figure()
plt.suptitle('Times of RL ' + network + ' models',y=1.0)
# ENV 2 - Reward
RLF_GET = []
RLF_STEP = []
RLF_UP = [] 
RLT_GET = []
RLT_STEP = []
RLT_UP = [] 
# False
with open(env2_RF,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLF_GET.append(float(row[0]))
        RLF_STEP.append(float(row[1]))
        RLF_UP.append(float(row[2]))

# Downsampled
with open(env2_RT,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLT_GET.append(float(row[0]))
        RLT_STEP.append(float(row[1]))
        RLT_UP.append(float(row[2]))

GET_std = [stat.pstdev(RLF_GET), stat.pstdev(RLT_GET)]
GET_mean = [stat.mean(RLF_GET), stat.mean(RLT_GET)]
print('GET False', GET_mean[0], '+-', GET_std[0])
print('GET True', GET_mean[1], '+-', GET_std[1])
STEP_std = [stat.pstdev(RLF_STEP), stat.pstdev(RLT_STEP)]
STEP_mean = [stat.mean(RLF_STEP), stat.mean(RLT_STEP)]
print('STEP False', STEP_mean[0], '+-', STEP_std[0])
print('STEP True', STEP_mean[1], '+-', STEP_std[1])
UP_std = [stat.pstdev(RLF_UP), stat.pstdev(RLT_UP)]
UP_mean = [stat.mean(RLF_UP), stat.mean(RLT_UP)]
print('UP False', UP_mean[0], '+-', UP_std[0])
print('UP True', UP_mean[1], '+-', UP_std[1])

plt.bar([0, 1], GET_mean, width, yerr=GET_std, label='Get action')
plt.bar([0, 1], STEP_mean, width, yerr=STEP_std, bottom=GET_mean, label='Environment step')
plt.bar([0, 1], UP_mean, width, yerr=UP_std, bottom=np.add(GET_mean, STEP_mean).tolist(), label='Update network')
plt.xticks([0, 1], ('Full graph', 'Downsampled graph'))
plt.ylabel('Time [s]')
#env1.title.set_text('Environment 1')#,y=1.25)
fig.text(0.5,0.08, "Environment 2", ha="center")
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=3)

plt.grid()
#plt.show()
fig.savefig('Figures/RL_times_' + network + '.eps', format='eps', dpi=1200,bbox_inches='tight')