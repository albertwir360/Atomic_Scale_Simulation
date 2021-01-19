import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.neighbors import KDTree
import matplotlib.lines as mlines


Folder = 'MaskAdherence1000/' #MaskAdherence
# Folder = 'SocialDistancing3000/' #SocialDistancing
# Folder = 'Gatherings3000/' #Gatherings


#Baseline Model Parameters
Social_distancing = False
Have_gatherings = False
pct_with_mask = 96 #0-100 is percentage
Quarantine_prob = 0.96 #0-1 is probability

if Folder == 'MaskAdherence1000/':
    # region RQ 1 Mask Adherence


    FullAveragedStates = []
    FullSDStates = []


    for pct_with_mask in [ 0. , 25. , 50. , 75. ,96. , 100. ]: #For Task 1 Mask Adherence

        CaseName = 'SD' + str(Social_distancing) + '-HG' + str(Have_gatherings) + '-m' + str(int(pct_with_mask)) + '-q' + str(Quarantine_prob) + '.npy'

        print(str(CaseName))

        States = np.load(Folder+CaseName)
        Replicates = np.shape(States)[0]
        StateCount = np.shape(States)[1]
        SimLength = np.shape(States)[2]

        #Averaged Table
        AveragedStates = np.zeros([StateCount,SimLength])
        SDStates = np.zeros([StateCount,SimLength])

        for i in range(0,SimLength):
            for j in range(0,StateCount):
                CellAvg = []

                for k in range(0,Replicates):
                    CellAvg.append(States[k][j][i])

                Mean = np.mean(CellAvg)
                SD = np.std(CellAvg)
                AveragedStates[j][i] = Mean*100/150
                SDStates[j][i] = SD*100/150
        FullAveragedStates.append(AveragedStates)
        FullSDStates.append(SDStates)

        # xtime = np.arange(0,SimLength)
        #
        # plt.plot(xtime,AveragedStates[0],label = 'Susceptible')
        # plt.fill_between(xtime,AveragedStates[0]-SDStates[0],AveragedStates[0]+SDStates[0],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[1],label = 'Exposed')
        # plt.fill_between(xtime,AveragedStates[1]-SDStates[1],AveragedStates[1]+SDStates[1],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[2],label = 'Infected')
        # plt.fill_between(xtime,AveragedStates[2]-SDStates[2],AveragedStates[2]+SDStates[2],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[3],label = 'Quarantined')
        # plt.fill_between(xtime,AveragedStates[3]-SDStates[3],AveragedStates[3]+SDStates[3],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[4],label = 'Removed')
        # plt.fill_between(xtime,AveragedStates[4]-SDStates[4],AveragedStates[4]+SDStates[4],alpha = 0.2)
        #
        #
        # # plt.errorbar(xtime,AveragedStates[0],yerr = SDStates[0],label = 'Susceptible')
        # # plt.errorbar(xtime,AveragedStates[1],yerr = SDStates[1],label = 'Exposed')
        # # plt.errorbar(xtime,AveragedStates[2],yerr = SDStates[2],label = 'Infected')
        # # plt.errorbar(xtime,AveragedStates[3],yerr = SDStates[3],label = 'Quarantined')
        # # plt.errorbar(xtime,AveragedStates[4],yerr = SDStates[4],label = 'Removed')
        # # plt.plot(AveragedStates[0]+AveragedStates[1]+AveragedStates[2]+AveragedStates[3]+AveragedStates[4],label = 'Sum')
        # plt.legend()
        # plt.title('Averaged State Counts')
        # plt.show()

        #IFR
        Fatality = np.load(Folder+'IFR_'+CaseName)

        #PNC
        PopNormCases = np.load(Folder+'PNC_'+CaseName)

        #Averaging Runs
        IFR = np.round(np.mean(Fatality),2)
        IFR_SD = np.round(np.std(Fatality),2)
        PNC = np.round(np.mean(PopNormCases),2)
        PNC_SD = np.round(np.std(PopNormCases),2)

        #PND
        PNDcalc = np.array(Fatality)*np.array(PopNormCases)/100
        PND = np.round(np.mean(PNDcalc),2)
        PND_SD = np.round(np.std(PNDcalc), 2)


        print('IFR is ' + str(IFR) + '%' + ' +/- ' + str(IFR_SD) + '%')
        print('PNC is ' + str(PNC) + '%' + ' +/- ' + str(PNC_SD) + '%')
        print('PND is ' + str(PND) + '%' + ' +/- ' + str(PND_SD) + '%')
        print(str(round(AveragedStates[0][-1],2))+'% Susceptible')
        print(str(round(AveragedStates[1][-1],2))+'% Exposed')
        print(str(round(AveragedStates[2][-1],2))+'% Infected')
        print(str(round(AveragedStates[3][-1],2))+'% Quarantined')
        print(str(round(AveragedStates[4][-1],2))+'% Removed')
        print('')

    FullAveragedStates = np.array(FullAveragedStates)
    FullSDStates = np.array(FullSDStates)

    xsteps = np.arange(0,np.shape(FullAveragedStates)[2])

    #All level figures
    # region Susceptible Cases
    plt.plot(xsteps,FullAveragedStates[0][0],label = '0% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][0],label = '25% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[2][0],label = '50% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[2][0]-FullSDStates[2][0],FullAveragedStates[2][0]+FullSDStates[2][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[3][0],label = '75% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[3][0]-FullSDStates[3][0],FullAveragedStates[3][0]+FullSDStates[3][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[4][0],label = '96% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[4][0]-FullSDStates[4][0],FullAveragedStates[4][0]+FullSDStates[4][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[5][0],label = '100% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[5][0]-FullSDStates[5][0],FullAveragedStates[5][0]+FullSDStates[5][0],alpha = 0.2)

    plt.title('Susceptible',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion
    # region Exposed Cases
    plt.plot(xsteps,FullAveragedStates[0][1],label = '0% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][1],label = '25% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[2][1],label = '50% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[2][0]-FullSDStates[2][0],FullAveragedStates[2][0]+FullSDStates[2][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[3][1],label = '75% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[3][0]-FullSDStates[3][0],FullAveragedStates[3][0]+FullSDStates[3][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[4][1],label = '96% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[4][0]-FullSDStates[4][0],FullAveragedStates[4][0]+FullSDStates[4][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[5][1],label = '100% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[5][0]-FullSDStates[5][0],FullAveragedStates[5][0]+FullSDStates[5][0],alpha = 0.2)

    plt.title('Exposed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18)
    plt.show()

    # endregion
    # region Infected Cases
    plt.plot(xsteps,FullAveragedStates[0][2],label = '0% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][2],label = '25% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[2][2],label = '50% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[2][0]-FullSDStates[2][0],FullAveragedStates[2][0]+FullSDStates[2][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[3][2],label = '75% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[3][0]-FullSDStates[3][0],FullAveragedStates[3][0]+FullSDStates[3][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[4][2],label = '96% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[4][0]-FullSDStates[4][0],FullAveragedStates[4][0]+FullSDStates[4][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[5][2],label = '100% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[5][0]-FullSDStates[5][0],FullAveragedStates[5][0]+FullSDStates[5][0],alpha = 0.2)

    plt.title('Infected',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion
    # region Quarantined Cases
    plt.plot(xsteps,FullAveragedStates[0][3],label = '0% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][3],label = '25% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[2][3],label = '50% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[2][0]-FullSDStates[2][0],FullAveragedStates[2][0]+FullSDStates[2][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[3][3],label = '75% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[3][0]-FullSDStates[3][0],FullAveragedStates[3][0]+FullSDStates[3][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[4][3],label = '96% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[4][0]-FullSDStates[4][0],FullAveragedStates[4][0]+FullSDStates[4][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[5][3],label = '100% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[5][0]-FullSDStates[5][0],FullAveragedStates[5][0]+FullSDStates[5][0],alpha = 0.2)

    plt.title('Quarantined',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion
    # region Removed Cases
    plt.plot(xsteps,FullAveragedStates[0][4],label = '0% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][4],label = '25% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[2][4],label = '50% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[2][0]-FullSDStates[2][0],FullAveragedStates[2][0]+FullSDStates[2][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[3][4],label = '75% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[3][0]-FullSDStates[3][0],FullAveragedStates[3][0]+FullSDStates[3][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[4][4],label = '96% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[4][0]-FullSDStates[4][0],FullAveragedStates[4][0]+FullSDStates[4][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[5][4],label = '100% Mask', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[5][0]-FullSDStates[5][0],FullAveragedStates[5][0]+FullSDStates[5][0],alpha = 0.2)

    plt.title('Removed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion

    #endregion

elif Folder == 'SocialDistancing3000/':
    # region RQ 2 Social Distancing


    FullAveragedStates = []
    FullSDStates = []
    for Social_distancing in [False, True]:

        CaseName = 'SD' + str(Social_distancing) + '-HG' + str(Have_gatherings) + '-m' + str(int(pct_with_mask)) + '-q' + str(Quarantine_prob) + '.npy'

        print(str(CaseName))

        States = np.load(Folder+CaseName)
        Replicates = np.shape(States)[0]
        StateCount = np.shape(States)[1]
        SimLength = np.shape(States)[2]

        #Averaged Table
        AveragedStates = np.zeros([StateCount,SimLength])
        SDStates = np.zeros([StateCount,SimLength])

        for i in range(0,SimLength):
            for j in range(0,StateCount):
                CellAvg = []

                for k in range(0,Replicates):
                    CellAvg.append(States[k][j][i])

                Mean = np.mean(CellAvg)
                SD = np.std(CellAvg)
                AveragedStates[j][i] = Mean*100/150
                SDStates[j][i] = SD*100/150
        FullAveragedStates.append(AveragedStates)
        FullSDStates.append(SDStates)

        # xtime = np.arange(0,SimLength)
        #
        # plt.plot(xtime,AveragedStates[0],label = 'Susceptible')
        # plt.fill_between(xtime,AveragedStates[0]-SDStates[0],AveragedStates[0]+SDStates[0],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[1],label = 'Exposed')
        # plt.fill_between(xtime,AveragedStates[1]-SDStates[1],AveragedStates[1]+SDStates[1],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[2],label = 'Infected')
        # plt.fill_between(xtime,AveragedStates[2]-SDStates[2],AveragedStates[2]+SDStates[2],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[3],label = 'Quarantined')
        # plt.fill_between(xtime,AveragedStates[3]-SDStates[3],AveragedStates[3]+SDStates[3],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[4],label = 'Removed')
        # plt.fill_between(xtime,AveragedStates[4]-SDStates[4],AveragedStates[4]+SDStates[4],alpha = 0.2)
        #
        #
        # # plt.errorbar(xtime,AveragedStates[0],yerr = SDStates[0],label = 'Susceptible')
        # # plt.errorbar(xtime,AveragedStates[1],yerr = SDStates[1],label = 'Exposed')
        # # plt.errorbar(xtime,AveragedStates[2],yerr = SDStates[2],label = 'Infected')
        # # plt.errorbar(xtime,AveragedStates[3],yerr = SDStates[3],label = 'Quarantined')
        # # plt.errorbar(xtime,AveragedStates[4],yerr = SDStates[4],label = 'Removed')
        # # plt.plot(AveragedStates[0]+AveragedStates[1]+AveragedStates[2]+AveragedStates[3]+AveragedStates[4],label = 'Sum')
        # plt.legend()
        # plt.title('Averaged State Counts')
        # plt.show()

        #IFR
        Fatality = np.load(Folder+'IFR_'+CaseName)

        #PNC
        PopNormCases = np.load(Folder+'PNC_'+CaseName)

        #Averaging Runs
        IFR = np.round(np.mean(Fatality),2)
        IFR_SD = np.round(np.std(Fatality),2)
        PNC = np.round(np.mean(PopNormCases),2)
        PNC_SD = np.round(np.std(PopNormCases),2)

        #PND
        PNDcalc = np.array(Fatality)*np.array(PopNormCases)/100
        PND = np.round(np.mean(PNDcalc),2)
        PND_SD = np.round(np.std(PNDcalc), 2)


        print('IFR is ' + str(IFR) + '%' + ' +/- ' + str(IFR_SD) + '%')
        print('PNC is ' + str(PNC) + '%' + ' +/- ' + str(PNC_SD) + '%')
        print('PND is ' + str(PND) + '%' + ' +/- ' + str(PND_SD) + '%')
        print(str(round(AveragedStates[0][-1],2))+'% Susceptible')
        print(str(round(AveragedStates[1][-1],2))+'% Exposed')
        print(str(round(AveragedStates[2][-1],2))+'% Infected')
        print(str(round(AveragedStates[3][-1],2))+'% Quarantined')
        print(str(round(AveragedStates[4][-1],2))+'% Removed')
        print('')

    FullAveragedStates = np.array(FullAveragedStates)
    FullSDStates = np.array(FullSDStates)

    xsteps = np.arange(0,np.shape(FullAveragedStates)[2])

    #All level figures
    # region Susceptible Cases
    plt.plot(xsteps,FullAveragedStates[0][0],label = 'SD False', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][0],label = 'SD True', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Susceptible',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    #endregion
    # region Exposed Cases
    plt.plot(xsteps,FullAveragedStates[0][1],label = 'SD False', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][1],label = 'SD True', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Exposed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18)
    plt.show()

     # endregion
    # region Infected Cases
    plt.plot(xsteps,FullAveragedStates[0][2],label = 'SD False', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][2],label = 'SD True', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Infected',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion
    # region Quarantined Cases
    plt.plot(xsteps,FullAveragedStates[0][3],label = 'SD False', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][3],label = 'SD True', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Quarantined',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    # endregion
    # region Removed Cases
    plt.plot(xsteps,FullAveragedStates[0][4],label = 'SD False', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][4],label = 'SD True', linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Removed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.show()

    #endregion
    #endregion

elif Folder == 'Gatherings3000/':
    # region RQ 3 Pods


    FullAveragedStates = []
    FullSDStates = []
    for Have_gatherings in [False, True]:

        CaseName = 'SD' + str(Social_distancing) + '-HG' + str(Have_gatherings) + '-m' + str(int(pct_with_mask)) + '-q' + str(Quarantine_prob) + '.npy'

        print(str(CaseName))

        States = np.load(Folder+CaseName)
        Replicates = np.shape(States)[0]
        StateCount = np.shape(States)[1]
        SimLength = np.shape(States)[2]

        #Averaged Table
        AveragedStates = np.zeros([StateCount,SimLength])
        SDStates = np.zeros([StateCount,SimLength])

        for i in range(0,SimLength):
            for j in range(0,StateCount):
                CellAvg = []

                for k in range(0,Replicates):
                    CellAvg.append(States[k][j][i])

                Mean = np.mean(CellAvg)
                SD = np.std(CellAvg)
                AveragedStates[j][i] = Mean*100/150
                SDStates[j][i] = SD*100/150
        FullAveragedStates.append(AveragedStates)
        FullSDStates.append(SDStates)

        # xtime = np.arange(0,SimLength)
        #
        # plt.plot(xtime,AveragedStates[0],label = 'Susceptible')
        # plt.fill_between(xtime,AveragedStates[0]-SDStates[0],AveragedStates[0]+SDStates[0],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[1],label = 'Exposed')
        # plt.fill_between(xtime,AveragedStates[1]-SDStates[1],AveragedStates[1]+SDStates[1],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[2],label = 'Infected')
        # plt.fill_between(xtime,AveragedStates[2]-SDStates[2],AveragedStates[2]+SDStates[2],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[3],label = 'Quarantined')
        # plt.fill_between(xtime,AveragedStates[3]-SDStates[3],AveragedStates[3]+SDStates[3],alpha = 0.2)
        #
        # plt.plot(xtime,AveragedStates[4],label = 'Removed')
        # plt.fill_between(xtime,AveragedStates[4]-SDStates[4],AveragedStates[4]+SDStates[4],alpha = 0.2)
        #
        #
        # # plt.errorbar(xtime,AveragedStates[0],yerr = SDStates[0],label = 'Susceptible')
        # # plt.errorbar(xtime,AveragedStates[1],yerr = SDStates[1],label = 'Exposed')
        # # plt.errorbar(xtime,AveragedStates[2],yerr = SDStates[2],label = 'Infected')
        # # plt.errorbar(xtime,AveragedStates[3],yerr = SDStates[3],label = 'Quarantined')
        # # plt.errorbar(xtime,AveragedStates[4],yerr = SDStates[4],label = 'Removed')
        # # plt.plot(AveragedStates[0]+AveragedStates[1]+AveragedStates[2]+AveragedStates[3]+AveragedStates[4],label = 'Sum')
        # plt.legend()
        # plt.title('Averaged State Counts')
        # plt.show()

        #IFR
        Fatality = np.load(Folder+'IFR_'+CaseName)

        #PNC
        PopNormCases = np.load(Folder+'PNC_'+CaseName)

        #Averaging Runs
        IFR = np.round(np.mean(Fatality),2)
        IFR_SD = np.round(np.std(Fatality),2)
        PNC = np.round(np.mean(PopNormCases),2)
        PNC_SD = np.round(np.std(PopNormCases),2)

        #PND
        PNDcalc = np.array(Fatality)*np.array(PopNormCases)/100
        PND = np.round(np.mean(PNDcalc),2)
        PND_SD = np.round(np.std(PNDcalc), 2)


        print('IFR is ' + str(IFR) + '%' + ' +/- ' + str(IFR_SD) + '%')
        print('PNC is ' + str(PNC) + '%' + ' +/- ' + str(PNC_SD) + '%')
        print('PND is ' + str(PND) + '%' + ' +/- ' + str(PND_SD) + '%')
        print(str(round(AveragedStates[0][-1],2))+'% Susceptible')
        print(str(round(AveragedStates[1][-1],2))+'% Exposed')
        print(str(round(AveragedStates[2][-1],2))+'% Infected')
        print(str(round(AveragedStates[3][-1],2))+'% Quarantined')
        print(str(round(AveragedStates[4][-1],2))+'% Removed')
        print('')

    FullAveragedStates = np.array(FullAveragedStates)
    FullSDStates = np.array(FullSDStates)

    xsteps = np.arange(0,np.shape(FullAveragedStates)[2])

    #All level figures
    # region Susceptible Cases

    plt.plot(xsteps,FullAveragedStates[0][0],label = 'HG False',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][0],label = 'HG True',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Susceptible',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18)
    plt.show()

    #endregion
    # region Exposed Cases
    plt.plot(xsteps,FullAveragedStates[0][1],label = 'HG False',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][1],label = 'HG True',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Exposed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    # plt.legend(fontsize = 18)
    plt.show()

     # endregion
    # region Infected Cases
    plt.plot(xsteps,FullAveragedStates[0][2],label = 'HG False',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][2],label = 'HG True',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Infected',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    # plt.legend(fontsize = 18)
    plt.show()

    # endregion
    # region Quarantined Cases
    plt.plot(xsteps,FullAveragedStates[0][3],label = 'HG False',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][3],label = 'HG True',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Quarantined',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    # plt.legend(fontsize = 18)
    plt.show()

    # endregion
    # region Removed Cases
    plt.plot(xsteps,FullAveragedStates[0][4],label = 'HG False',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[0][0]-FullSDStates[0][0],FullAveragedStates[0][0]+FullSDStates[0][0],alpha = 0.2)

    plt.plot(xsteps,FullAveragedStates[1][4],label = 'HG True',linewidth = 3)
    # plt.fill_between(xsteps,FullAveragedStates[1][0]-FullSDStates[1][0],FullAveragedStates[1][0]+FullSDStates[1][0],alpha = 0.2)

    plt.title('Removed',fontsize = 18)
    plt.xlabel('Timesteps',fontsize = 18)
    plt.ylabel('Population %',fontsize = 18)
    plt.tick_params(labelsize = 18)
    # plt.legend(fontsize = 18)
    plt.show()

    #endregion
    #endregion

else:
    print('Diretory Not Specified Correctly')

