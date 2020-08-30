import matplotlib.pyplot as plt
import csv
import statistics as stat
import numpy as np
import os

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

network = 'PNA'
os.makedirs('Figures/' + network + '/', exist_ok = True)

if (network == 'DQN'):
    times = ['2020-08-29-12', '2020-08-29-19']
elif (network == 'GAT'):
    times = ['2020-08-29-13', '2020-08-29-19']
elif (network == 'GGNN'):
    times = ['2020-08-29-15', '2020-08-29-21']
elif (network == 'NN'):
    times = ['2020-08-29-17', '2020-08-29-21']
elif (network == 'SAGE'):
    times = ['2020-08-29-18', '2020-08-29-22']
elif (network == 'GCN'):
    times = ['2020-08-29-13', '2020-08-29-20']
elif (network == 'CG'):
    times = ['2020-08-29-14', '2020-08-29-20']
elif (network == 'SGN'):
    times = ['2020-08-29-14', '2020-08-29-20']
elif (network == 'GIN'):
    times = ['2020-08-29-15', '2020-08-29-21']
elif (network == 'PNA'):
    times = ['2020-08-29-18', '2020-08-29-22']

env2_RF = 'Results/' + network + '/' + network + '_' + times[0] + '.csv'
env2_RT = 'Results/' + network + '/' + network + '_' + times[1] + '.csv'

fig = plt.figure()
plt.suptitle('Time comparison of ' + network,y=1.0)
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

GET_std = [stat.pstdev(RLF_GET)*1000, stat.pstdev(RLT_GET)*1000]
GET_mean = [stat.mean(RLF_GET)*1000, stat.mean(RLT_GET)*1000]
print('GET False', round(GET_mean[0],3), '+-', round(GET_std[0],3))
print('GET True', round(GET_mean[1],3), '+-', round(GET_std[1],3))
STEP_std = [stat.pstdev(RLF_STEP)*1000, stat.pstdev(RLT_STEP)*1000]
STEP_mean = [stat.mean(RLF_STEP)*1000, stat.mean(RLT_STEP)*1000]
print('STEP False', round(STEP_mean[0],3), '+-', round(STEP_std[0],3))
print('STEP True', round(STEP_mean[1],3), '+-', round(STEP_std[1],3))
UP_std = [stat.pstdev(RLF_UP)*1000, stat.pstdev(RLT_UP)*1000]
UP_mean = [stat.mean(RLF_UP)*1000, stat.mean(RLT_UP)*1000]
print('UP False', round(UP_mean[0],3), '+-', round(UP_std[0],3))
print('UP True', round(UP_mean[1],3), '+-', round(UP_std[1],3))
Total_mean = [GET_mean[0]+STEP_mean[0]+UP_mean[0], GET_mean[1]+STEP_mean[1]+UP_mean[1]]
Total_std = [GET_std[0]+STEP_std[0]+UP_std[0], GET_std[1]+STEP_std[1]+UP_std[1]]
print('Total False', round(Total_mean[0],3), '+-', round(Total_std[0],3))
print('Total True', round(Total_mean[1],3), '+-', round(Total_std[1],3))
print('Speedup', round(Total_mean[0]/Total_mean[1],3))

plt.bar([0, 1], GET_mean, width, label='Get action')
plt.bar([0, 1], STEP_mean, width, bottom=GET_mean, label='Environment step')
plt.bar([0, 1], UP_mean, width, bottom=np.add(GET_mean, STEP_mean).tolist(), label='Update network')
plt.xticks([0, 1], ('Full graph', 'Downsampled graph'))
plt.ylabel('Time [$\mu s$]')
#env1.title.set_text('Environment 1')#,y=1.25)
fig.text(0.5,0.08, "Environment 2", ha="center")
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=3)

plt.grid()
#plt.show()
fig.savefig('Figures/' + network + '/RL_times_' + network + '.eps', format='eps', dpi=1200,bbox_inches='tight')