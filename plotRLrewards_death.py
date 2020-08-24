import matplotlib.pyplot as plt
import csv
#markerstyle= (cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', ['^',',', '.']))
#
#plt.rcParams['axes.grid'] = True
##plt.rcParams['axes.prop_cycle'] = markerstyle
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
    times_F = ['2020-08-12-16', '2020-07-31-20', '2020-08-02-22', '2020-08-12-00', '2020-08-13-13']
    times_T = ['2020-07-29-09', '2020-07-31-09', '2020-08-02-16', '2020-08-12-10', '2020-08-16-21']
elif (network == 'GAT'):
    times_F = ['2020-08-12-16', '2020-07-31-20', '2020-08-03-09', '2020-08-04-22', '2020-08-13-13']
    times_T = ['2020-07-29-09', '2020-07-31-11', '2020-08-02-16', '2020-08-04-16', '2020-08-18-17']
elif (network == 'GGNN'):
    times_F = ['2020-07-29-22', '2020-07-30-08', '2020-08-01-13', '2020-08-12-00', '2020-08-13-13']
    times_T = ['2020-07-29-16', '2020-07-30-17', '2020-08-02-09', '2020-08-05-08', '2020-08-16-21']
elif (network == 'NN'):
    times_F = ['2020-07-29-22', '2020-07-30-08', '2020-08-01-13', '2020-08-12-10', '2020-08-13-13']
    times_T = ['2020-07-29-16', '2020-07-30-17', '2020-08-02-09', '2020-08-05-08', '2020-08-16-21']
elif (network == 'SAGE'):
    times_F = ['2020-07-30-08', '2020-08-12-15', '2020-08-01-13', '2020-08-04-16', '2020-08-13-13']
    times_T = ['2020-07-29-16', '2020-08-03-09', '2020-08-02-09', '2020-08-05-08', '2020-08-19-22']
elif (network == 'GCN'):
    times_F = ['2020-08-12-16', '2020-07-31-20', '2020-08-03-09', '2020-08-04-22', '2020-08-18-17']
    times_T = ['2020-07-29-09', '2020-07-31-11', '2020-08-02-16', '2020-08-04-16', '2020-08-22-10']
elif (network == 'CG'):
    times_F = ['2020-08-13-00', '2020-07-31-20', '2020-08-02-22', '2020-08-12-00', '2020-08-18-17']
    times_T = ['2020-07-29-09', '2020-07-31-09', '2020-08-02-16', '2020-08-12-10', '2020-08-21-09']
elif (network == 'SGN'):
    times_F = ['2020-08-13-00', '2020-07-31-20', '2020-08-02-22', '2020-08-12-00', '2020-08-19-11']
    times_T = ['2020-07-29-09', '2020-07-31-09', '2020-08-02-16', '2020-08-12-10', '2020-08-22-12']
elif (network == 'GIN'):
    times_F = ['2020-07-29-22', '2020-07-30-08', '2020-08-01-13', '2020-08-12-00', '2020-08-22-20']
    times_T = ['2020-07-29-16', '2020-07-30-17', '2020-08-02-09', '2020-08-05-09', '']
elif (network == 'PNA'):
    times_F = ['2020-07-29-22', '2020-07-30-08', '2020-08-01-13', '2020-08-12-10', '2020-08-21-15']
    times_T = ['2020-07-29-16', '2020-07-30-17', '2020-08-02-09', '2020-08-05-08', '2020-08-24-16']

env2_30F = 'Results/'+ network + '/Test/' + times_F[0] + '/NumToteTest_' + times_F[0] + '.csv'
env2_50F = 'Results/'+ network + '/Test/' + times_F[1] + '/NumToteTest_' + times_F[1] + '.csv'
env2_80F = 'Results/'+ network + '/Test/' + times_F[2] + '/NumToteTest_' + times_F[2] + '.csv'
env2_R1F = 'Results/'+ network + '/Test/' + times_F[3] + '/NumToteTest_' + times_F[3] + '.csv'

fig = plt.figure()
plt.suptitle('Reward of RL ' + network + ' models',y=1.02)
# ENV 2 - Reward
x = []
RL30 = []
RL50 = []
RL80 = []
RLR = []
SSP = []
DSP = [] 
# SSP, DSP, RL30%
with open(env2_30F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL30.append(float(row[8]))
        SSP.append(float(row[9]))
        DSP.append(float(row[10]))
        x.append(int(100*float(row[1])))

# RL50%
with open(env2_50F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL50.append(float(row[8]))

# RL80%
with open(env2_80F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL80.append(float(row[8]))
        
# RLR%
with open(env2_R1F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLR.append(float(row[8]))
        
#env2 = fig.add_subplot(3,1,1)
markevery_=1
plt.plot(x, RL30,',--', fillstyle='none', label='RL30',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RL50,'.--', fillstyle='none', label='RL50',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RL80,'^-', fillstyle='none', label='RL80',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RLR, 'k.-', fillstyle='none', label='RLR',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, SSP, '^-', fillstyle='none', label='SSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, DSP, '*-', fillstyle='none', label='DSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.xlabel('Load')
plt.ylabel('Average reward')
#env1.title.set_text('Environment 1')#,y=1.25)
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)

plt.grid()
#plt.show()
#env1.savefig('env1-reward.eps', format='eps', dpi=1200,bbox_inches='tight')
fig.savefig('Figures/RL_reward_' + network + '_F.eps', format='eps', dpi=1200,bbox_inches='tight')

fig = plt.figure()
plt.suptitle('Deathlock of RL ' + network + ' models',y=1.02)
# ENV 2 - Deadlocks
x = []
RL30 = []
RL50 = []
RL80 = []
RLR = []
SSP = []
DSP = []
# SSP, DSP, RL30%
with open(env2_30F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',',markevery=5)
    for row in plots:
        RL30.append(float(row[14]))
        SSP.append(float(row[15]))
        DSP.append(float(row[16]))
        x.append(int(100*float(row[1])))

# RL50%
with open(env2_50F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL50.append(float(row[14]))

# RL80%
with open(env2_80F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL80.append(float(row[14]))
        
# RLR%
with open(env2_R1F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLR.append(float(row[14]))
        
plt.plot(x, RL30,',-', label='RL30')
plt.plot(x, RL50,'.-', label='RL50')
plt.plot(x, RL80,',--', label='RL80')
plt.plot(x, RLR, '.--', label='RLR')
plt.plot(x, SSP, '^:', label='SSP')
plt.plot(x, DSP, ',:', label='DSP')
plt.xlabel('Load')
plt.ylabel('Deadlocks in 100 iterations')
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)
plt.grid()
#plt.show()
fig.savefig('Figures/RL_deathlock_' + network + '_F.eps', format='eps', dpi=1200)

env2_30T = 'Results/'+ network + '/Test/' + times_T[0] + '/NumToteTest_' + times_T[0] + '.csv'
env2_50T = 'Results/'+ network + '/Test/' + times_T[1] + '/NumToteTest_' + times_T[1] + '.csv'
env2_80T = 'Results/'+ network + '/Test/' + times_T[2] + '/NumToteTest_' + times_T[2] + '.csv'
env2_R1T = 'Results/'+ network + '/Test/' + times_T[3] + '/NumToteTest_' + times_T[3] + '.csv'

fig = plt.figure()
plt.suptitle('Reward of RL ' + network + ' models - Downsampled',y=1.02)
# ENV 2 - Reward
x = []
RL30 = []
RL50 = []
RL80 = []
RLR = []
SSP = []
DSP = [] 
# SSP, DSP, RL30%
with open(env2_30T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL30.append(float(row[8]))
        SSP.append(float(row[9]))
        DSP.append(float(row[10]))
        x.append(int(100*float(row[1])))

# RL50%
with open(env2_50T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL50.append(float(row[8]))

# RL80%
with open(env2_80T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL80.append(float(row[8]))
        
# RLR%
with open(env2_R1T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLR.append(float(row[8]))
        
#env2 = fig.add_subplot(3,1,1)
markevery_=1
plt.plot(x, RL30,',--', fillstyle='none', label='RL30',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RL50,'.--', fillstyle='none', label='RL50',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RL80,'^-', fillstyle='none', label='RL80',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RLR, 'k.-', fillstyle='none', label='RLR',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, SSP, '^-', fillstyle='none', label='SSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, DSP, '*-', fillstyle='none', label='DSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.xlabel('Load')
plt.ylabel('Average reward')
#env1.title.set_text('Environment 1')#,y=1.25)
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)

plt.grid()
#plt.show()
#env1.savefig('env1-reward.eps', format='eps', dpi=1200,bbox_inches='tight')
fig.savefig('Figures/RL_reward_' + network + '_T.eps', format='eps', dpi=1200,bbox_inches='tight')

fig = plt.figure()
plt.suptitle('Deathlock of RL ' + network + ' models - Downsampled',y=1.02)
# ENV 2 - Deadlocks
x = []
RL30 = []
RL50 = []
RL80 = []
RLR = []
SSP = []
DSP = []
# SSP, DSP, RL30%
with open(env2_30T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',',markevery=5)
    for row in plots:
        RL30.append(float(row[14]))
        SSP.append(float(row[15]))
        DSP.append(float(row[16]))
        x.append(int(100*float(row[1])))

# RL50%
with open(env2_50T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL50.append(float(row[14]))

# RL80%
with open(env2_80T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL80.append(float(row[14]))
        
# RLR%
with open(env2_R1T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RLR.append(float(row[14]))
        
plt.plot(x, RL30,',-', label='RL30')
plt.plot(x, RL50,'.-', label='RL50')
plt.plot(x, RL80,',--', label='RL80')
plt.plot(x, RLR, '.--', label='RLR')
plt.plot(x, SSP, '^:', label='SSP')
plt.plot(x, DSP, ',:', label='DSP')
plt.xlabel('Load')
plt.ylabel('Deadlocks in 100 iterations')
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)
plt.grid()
#plt.show()
fig.savefig('Figures/RL_deathlock_' + network + '_T.eps', format='eps', dpi=1200)

env2_R10F = 'Results/'+ network + '/Test/' + times_F[4] + '/NumToteTest_' + times_F[4] + '.csv'
env2_R10T = 'Results/'+ network + '/Test/' + times_T[4] + '/NumToteTest_' + times_T[4] + '.csv'

fig = plt.figure()
plt.suptitle('Reward of RL ' + network + ' models - Long Training',y=1.02)
# ENV 2 - Reward
x = []
RL10F = []
RL10T = []
SSP = []
DSP = [] 
# SSP, DSP, RL10F%
with open(env2_R10F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL10F.append(float(row[8]))
        SSP.append(float(row[9]))
        DSP.append(float(row[10]))
        x.append(int(100*float(row[1])))

# RL10T%
with open(env2_R10T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL10T.append(float(row[8]))
        
#env2 = fig.add_subplot(3,1,1)
markevery_=1
plt.plot(x, RL10F,',--', fillstyle='none', label='RLF',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, RL10T,'.--', fillstyle='none', label='RLT',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, SSP, '^-', fillstyle='none', label='SSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.plot(x, DSP, '*-', fillstyle='none', label='DSP',markevery=markevery_,lw=lw,ms=ms,mew=lw)
plt.xlabel('Load')
plt.ylabel('Average reward')
#env1.title.set_text('Environment 1')#,y=1.25)
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)

plt.grid()
#plt.show()
#env1.savefig('env1-reward.eps', format='eps', dpi=1200,bbox_inches='tight')
fig.savefig('Figures/RL_reward_' + network + '_Long.eps', format='eps', dpi=1200,bbox_inches='tight')

fig = plt.figure()
plt.suptitle('Deathlock of RL ' + network + ' models - Long Training',y=1.02)
# ENV 2 - Deadlocks
x = []
RL10F = []
RL10T = []
SSP = []
DSP = []
# SSP, DSP, RL10F%
with open(env2_R10F,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',',markevery=5)
    for row in plots:
        RL10F.append(float(row[14]))
        SSP.append(float(row[15]))
        DSP.append(float(row[16]))
        x.append(int(100*float(row[1])))

# RLR%
with open(env2_R10T,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        RL10T.append(float(row[14]))
      
plt.plot(x, RL10F,',--', label='RLF')
plt.plot(x, RL10T, '.--', label='RLT')
plt.plot(x, SSP, '^:', label='SSP')
plt.plot(x, DSP, ',:', label='DSP')
plt.xlabel('Load')
fig.text(0.5,-0.25, "Environment 2", ha="center", transform=fig.transAxes)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.6), ncol=3)
plt.grid()
#plt.show()
fig.savefig('Figures/RL_deathlock_' + network + '_Long.eps', format='eps', dpi=1200)
