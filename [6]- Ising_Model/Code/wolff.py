import numpy as np
import statistics
import math

# initialize random number generator
# note: does not work on old versions of numpy
# if so, then replace all the RNG calls with something else
RNG = np.random.default_rng()
def my_std(a):
    return (statistics.stdev(a))
    
# def my_actime(a): 
#     norm = 1/np.sqrt((np.size(a)-1))
#     f_sum = 0
#     for i in a:
#         f_sum = f_sum + i*i
#     s_sum = (1/np.size(a))*(np.sum(a))*(np.sum(a))
    return norm*np.sqrt(f_sum-s_sum)
def c_func(a,t):
    N = len(a)
    mean = np.mean(a)
    std = my_std(a)
    c_sum = 0
    for i in range(0,N-t):
        c_sum += (a[i]-mean)*(a[i+t]-mean)
    return (1/np.square(std))*(1/(N-t))* c_sum
def my_actime(a): 
    t_cutoff = 0
    for i in range (1,len(a)):
        if c_func(a,i) <= 0:
            t_cutoff = i
            break
    k_sum = 0
    for i in range(1, t_cutoff):
        k_sum += c_func(a,i)
    return 1+2*k_sum

def my_stderr(a):
    
    std = my_std(a)
    N = len(a)
    autocorr = my_actime(a)
    standard_error = std / math.sqrt(N/autocorr)
    return standard_error
def my_neighbor_list(i, j, N):
  """ Find all neighbors of site (i, j).

  Args:
    i (int): site index along x
    j (int): site index along y
    N (int): number of sites along each dimension
  Return:
    list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above),
     (i_right, j_right), (i_below, j_below)]
  """
  # fix these indices
  left   = ((i-1)%N, j)
  above  = (i, (j+1)%N)
  right  = ((i+1)%N, j)
  below  = (i, (j-1)%N)
  return [left, above, right, below]

def my_energy_site(i, j, lattice):
  """ Calculate energy associated with site (i, j)

  The sum of energy_site over all sites should be 2* total energy.

  Args:
    i (int): site index along x
    j (int): site index along y
    lattice (np.array): shape (N, N), 2D lattice of +1 and -1
  Return:
    float: energy of site (i, j)
  """
  N = lattice.shape[0]
  energy = 0
  a,b,c,d = my_neighbor_list(i, j, len(lattice))
  energy = lattice[a[0],a[1]] + lattice[b[0],b[1]] +lattice[c[0],c[1]] + lattice[d[0],d[1]]
  return energy
  
def my_spin_flip_random(lattice, i, j, beta):
  """ Calculate spin flip probability and change to total magnetization.

  Args:
    lat (np.array): shape (N, N), 2D lattice of +1 and -1
    i (int): site index along x
    j (int): site index along y
    beta (float): inverse temperature
  Return:
    (float, int): (A, dM). A is acceptance ratio. dM is change to
     total magnetization due to the spin flip.
  """
  de = -my_energy_site(i, j, lattice) + my_energy_site(i, j, -1 * lattice)
  dM = -lattice[i,j]*2
  
  if dM < 0 :
      de = -de
      lattice[i,j] = -lattice[i,j]
  A = np.exp(-beta * de)
  return A, dM

def neighbor_indices(site, N):
    """
    returns a list of tuples,
    where each tuple is a pair of indices that accesses
    one of the four neighbors to site (i, j)
    
    even though python accepts negative indices,
    it does make my implementation of cluster_flip() trickier
    if negative indices are allowed

    the reason for using tuples is the following:
    if x is a 2D array, x[(i,j)] = x[i,j]
    BUT x[[i,j]] is different; i want the former

    args:
        site (tuple): pair of indices (i, j) to access site
        N (int): number of sites along one direction,
        or the length of the array of spins
    returns:
        list of tuples, length 4, where each tuple has length 2
        and contains positive integer indices
        to access each of the four neighbors
    """
    i = site[0]
    j = site[1]
    right = ((i + 1)%N, j)
    left = ((i - 1 + N)%N, j)
    up = (i, (j + 1)%N)
    down = (i, (j - 1 + N)%N)
    return [right, left, up, down]

def cluster_flip(s, i, j, p):
    """
    the Wolff algorithm, beginning with the spin on site i,j

    args:
        s (array): lattice of spins,
        i.e. is a square array of 1's and -1's
        i (int): first index of seed spin
        j (int): second index of seed spin
        p (float): addition probability
    returns:
        array of the same shape as s,
        which is s after the cluster has been flipped
    """
    lattice = np.copy(s) # work on a copy to avoid any funny business
    N = s.shape[0]
    added = np.zeros_like(s, dtype=bool)
    indices = []
    # the strategy is the following:
    # added[i,j] is true if lattice[i,j] is added to the cluster
    # furthermore, the pair (i,j) will be stored in indices list
    added[i,j] = True
    indices.append((i,j))
    count = 0 # to traverse indices
    # begin adding
    while (count < len(indices)):
        neighbors = neighbor_indices(indices[count], N)
        for neighbor in neighbors:
            parallel = (lattice[indices[count]]*lattice[neighbor] > 0)
            if (parallel and not added[neighbor]):
                # consider addition to the cluster
                if RNG.random() < p:
                    indices.append(neighbor)
                    added[neighbor] = True
        count += 1
    # flip the cluster
    for i in indices:
        lattice[i] *= -1
    return lattice

# create system
N = 20
spins = RNG.integers(0, 2, size=(N,N))
spins[spins==0] = -1
temperature = 2.6
addition_probability = 1 - np.exp(-2./temperature)
number_of_samples = 10000
energy_array = []
magnetization = []
for _ in range(number_of_samples):
    # print(_)
    i = RNG.integers(0, N)
    j = RNG.integers(0, N)
    spins = cluster_flip(spins, i, j, addition_probability)
    # A = my_spin_flip_random(spins,i,j,1/temperature)
    # r = np.random.uniform()
    # print(A)
    # print(r)
    # if A[1] > r:
    #     spins[i,j] = -spins[i,j]
    # put your measurements here
    for i in range(N):
        Esum = 0
        Msum = 0
        for j in range(N):
            Esum += my_energy_site(i,j,spins)/2
            Msum += spins[i,j]
        energy_array.append(Esum)
        magnetization.append((Msum/(N**2))**2)
print(np.average(energy_array)/N**2)
print(np.average(magnetization))
print(my_actime(magnetization))
# print(my_stderr(energy_array))
# print(my_stderr(magnetization))
