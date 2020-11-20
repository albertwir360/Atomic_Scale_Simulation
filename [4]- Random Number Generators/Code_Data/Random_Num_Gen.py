import numpy as np 
from matplotlib import pyplot as plt
import math

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
def my_LCG(m,a,c,x0,N):
    """
    Copy your LCG over here
    """
    array = []
    x = x0
    for i in range(N):
        x = (a * x + c) % m
        array.append(x)
    
    return array
    pass

def my_GaussianRNG(m,a,c,x0,N):
    """
    Copy your GaussianRNG over here
    """
    x = my_LCG(m,a,c,x0, 2*N)
    array = []
    for i in range(0, int(N)):
        the_log = np.sqrt(-2 * np.log(x[2*i]/m))
        the_sin = np.sin(2* np.pi * x[2*i+1]/m)
        array.append(the_log *the_sin)
    return array


def my_ComputeIntegral2(func, alpha, N):
    """
    Input:
    func: a well defined function that decays fast when |x| goes to infinity
    alpha: variance of the normal distribution to be sampled
    N: length of Gaussian random numbers

    Output:
    a two-element list or numpy array, with the first element being the estimate of the integral,
    and the second being the the estimate of the variance
    """
    #-------------------------------------------------------------------
    # We will be using m=2^32, a=69069, c=1, and x0=0 for my_GaussianRNG
    # Multiply the stream of Gaussian random numbers by np.sqrt(alpha) to make their variance equal to alpha
    gaussian_arrays = np.sqrt(alpha) * np.array(my_GaussianRNG(2**32, 69069, 1, 0, N))
    #-------------------------------------------------------------------
    # define p(x, alpha)
    def p(x,alpha):
        return np.exp(-x*x/(2.0*alpha)) / np.sqrt(2.0*np.pi*alpha)
    #-------------------------------------------------------------------
    # complete this function
    mean = 0
    variance = 0
    g_sum = 0 
    for i in gaussian_arrays:
        g_sum += func(i)/ p(i,alpha)
    mean = g_sum/len(gaussian_arrays)
    v_sum = 0
    for i in gaussian_arrays:
        v_sum += np.square(func(i)/p(i,alpha) - mean)
    variance = v_sum/len(gaussian_arrays)
        
    return variance

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

def my_LCG(m,a,c,x0,N):
    """
    Input:
    m, a, c are parameters of the LCG.
    x0: the initial pseduo-random number.
    N : number of random numbers to return

    Output:
    a list or numpy array of length N, containing the next N pseduo-random numbers in order (excluding x0).
    """
    array = []
    x = x0
    for i in range(N):
        x = (a * x + c) % m
        array.append(x)
    
    return array
    
m = 2**32
a = 69069
c = 1
x0 = 0
N= 10000

def my_GaussianRNG(m,a,c,x0,N):
    """
    Input:
    m, a, c, x0 are parameters of the LCG.
    N : number of Gaussian random numbers to return

    Output:
    a list or numpy array of length N, containing the next N Gaussian pseduo-random numbers in order (excluding x0).
    """
    x = my_LCG(m,a,c,x0, 2*N)
    array = []
    for i in range(0, int(N)):
        the_log = np.sqrt(-2 * np.log(x[2*i]/m))
        the_sin = np.sin(2* np.pi * x[2*i+1]/m)
        array.append(the_log *the_sin)
    return array
def calc_dx(a,b,N):
    return (b-a)/float(N)
    
def my_ComputeIntegral(func, L, N):
    """
    Input:
    func: function with a single input f(x)
    L: sets the bounds of the integral by [-L, L]
    N: number of rectangles

    Output:
    the integral using rectangle rule
    """
    a = -L
    b = L
    total = 0.0 
    dx = calc_dx(a,b,N)
    for i in range(0,N+1):
        total += func((a + (i*dx)))
    return total*dx

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    # plt.plot(x, y, color = 'green')  

bab = my_GaussianRNG(2**32, 69069, 1, 0, 10000)
histo = np.histogram(bab, bins=100, density=True)
hist, bins = histo
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot()

# graph('np.exp(-x**2/2)/np.sqrt(2*np.pi)', np.arange(-5.0, 5.0, 0.1))
# plt.title('Obtained histogram compared to Theoretical Function')
# plt.show()


random1 = my_LCG(2**32, 69069,1,0,6000)
random2 = np.array(my_LCG(2**32, 69069,1,0,6000)).reshape(3000,2)
random3 =  np.array(my_LCG(2**32, 69069,1,0,6000)).reshape(2000,3)

randrand1 = np.random.random((6000))
randrand2 = randrand1.reshape(3000,2)
randrand3 = randrand1.reshape(2000, 3)

print(my_CheckRandomNumbers1D(random1, 10))
print(my_CheckRandomNumbers2D(random2, 10))
print(my_CheckRandomNumbers1D(random3, 10))

print(my_CheckRandomNumbers1D(randrand1, 10))
print(my_CheckRandomNumbers1D(randrand2, 10))
print(my_CheckRandomNumbers1D(randrand3, 10))
print(random2)
print(random2[0])
print(random2[0][0])
print(random2[0][1])

new = np.array(my_LCG(2**32, 5,1,0,6000)).reshape(3000,2)

xx = []
yy = []
for i in range(3000):
    xx.append(new[i][0])
    yy.append(new[i][1])

randrand1 = np.random.random((6000))
randrand2 = randrand1.reshape(3000,2)

xxx = []
yyy = []
for i in range(3000):
    xxx.append(randrand2[i][0])
    yyy.append(randrand2[i][1])
# plt.scatter(xx, yy, cmap='virdis')
# plt.show()


def func(x):
    return math.exp(-x**2/2)/(1+x**2)

# my_ComputeIntegral(func, 5, 1000)

# print(my_ComputeIntegral2(func, 0.1, 1000))
# print(my_ComputeIntegral2(func, 0.15, 1000))
# print(my_ComputeIntegral2(func, 0.2, 1000))
# print(my_ComputeIntegral2(func, 0.25, 1000))
# print(my_ComputeIntegral2(func, 0.3, 1000))
# print(my_ComputeIntegral2(func, 0.4, 1000))
# print(my_ComputeIntegral2(func, 0.5, 1000))
# print(my_ComputeIntegral2(func, 0.51, 1000))
# print(my_ComputeIntegral2(func, 0.52, 1000))
# print(my_ComputeIntegral2(func, 0.53, 1000))
# print(my_ComputeIntegral2(func, 0.54, 1000))
# print(my_ComputeIntegral2(func, 0.6, 1000))
# print(my_ComputeIntegral2(func, 0.7, 1000))
# print(my_ComputeIntegral2(func, 0.8, 1000))
# print(my_ComputeIntegral2(func, 0.9, 1000))
# x = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.51, 0.52, 0.53, 0.54, 0.6, 0.7, 0.8, 0.9]
# y = [my_ComputeIntegral2(func, 0.1, 1000),
# my_ComputeIntegral2(func, 0.15, 1000),
# my_ComputeIntegral2(func, 0.2, 1000),
# my_ComputeIntegral2(func, 0.25, 1000),
# my_ComputeIntegral2(func, 0.3, 1000),
# my_ComputeIntegral2(func, 0.4, 1000),
# my_ComputeIntegral2(func, 0.5, 1000),
# my_ComputeIntegral2(func, 0.51, 1000),
# my_ComputeIntegral2(func, 0.52, 1000),
# my_ComputeIntegral2(func, 0.53, 1000),
# my_ComputeIntegral2(func, 0.54, 1000),
# my_ComputeIntegral2(func, 0.6, 1000),
# my_ComputeIntegral2(func, 0.7, 1000),
# my_ComputeIntegral2(func, 0.8, 1000),

# my_ComputeIntegral2(func, 0.9, 1000)]



the_x = []
the_y = []
for i in np.arange(0.1, 1.0, 0.01):
    x = my_ComputeIntegral2(func, i, 1000)
    the_y.append(x)
    the_x.append(i)
plt.plot(the_x,the_y)
plt.show()










