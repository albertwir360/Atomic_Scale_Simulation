import numpy as np

def my_CheckRandomNumbers1D(rand_array,NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of length N
    NB: number of bins per dimension (for 1D we need NB bins in total)

    Output:
    the chi-squared value of the rand_array, with NB evenly spaced bins in [0,1).
    """
    # complete this function
    histo = np.histogram(rand_array,bins =NB, range = (0,1))
    chi = 0 
    expected = len(rand_array)/ NB 
    for i in range(NB):
        val = np.square(histo[0][i]-expected)/expected
        chi+= val
    return chi
def my_CheckRandomNumbers3D(rand_array,NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of size N-by-3, so (rand_array[0][0], rand_array[0][1], rand_array[0][2]) is the first 3D point
    NB: number of bins per dimension (for 3D we need NB*NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array, using NB*NB*NB evenly spaced bins in [0,1)x[0,1)x[0,1).
    """
    # complete this function
    sum = 0.0
    hist = np.histogramdd(rand_array, bins = NB, range = [[0,1], [0,1], [0,1]])
    aveBin = len(rand_array)/NB**3
    for i in range(0, NB):
        sum += ((hist[0][i]- aveBin)**2)/ aveBin
    return np.sum(sum)

def my_CheckRandomNumbers2D(rand_array,NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of size N-by-2, so (rand_array[0][0], rand_array[0][1]) is the first 2D point
    NB: number of bins per dimension (for 2D we need NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array, using NB*NB evenly spaced bins in [0,1)x[0,1).
    """
    # complete this function
    sum = 0.0
    x =[]
    y = []
    for i in range(len(rand_array)):
        x.append(rand_array[i][0])
        y.append(rand_array[i][1])
    hist = np.histogram2d(x,y,bins = NB, range =[[0,1],[0,1]])
    aveBin = len(rand_array)/NB**2
    
    for i in range(0,NB):
        sum += ((hist[0][i]-aveBin)**2)/(aveBin)
    return np.sum(sum)

a = np.loadtxt('1D_chi_squared_data_set_1.txt')

print(my_CheckRandomNumbers1D(a,100))

b = np.loadtxt('2D_chi_squared_data_set_1.txt')
print(my_CheckRandomNumbers2D(b,10))

c = np.loadtxt('3D_chi_squared_data_set_1.txt')
print(my_CheckRandomNumbers3D(c,10))
