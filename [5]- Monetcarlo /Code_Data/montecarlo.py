import numpy as np
import random 
import math
import time 
import statistics



# def minimum_image(r, L):
#     """
#     required for: displacement_table(), advance()
#     args:
#         r: array of any shape
#         L: side length of cubic box
#     returns:
#         array of the same shape as r,
#         the minimum image of r
#     """
#     given_pos = np.copy(r)
#     for i in range(0, len(r)):
#         if given_pos[i] > 0:
#             given_pos[i] = given_pos[i] - L * np.ceil((given_pos[i] - L / 2) / L)
#         elif given_pos[i] <= 0:
#             given_pos[i] = given_pos[i] + L * np.ceil((-given_pos[i] - L / 2) / L)
#     return given_pos
#     pass
def minimum_image(r, L):
  return r - L*np.round(r / L)


def cubic_lattice(tiling, L):
    """
    required for: initialization

    args:
        tiling (int): determines number of coordinates,
        by tiling^3
        L (float): side length of simulation box
    returns:
        array of shape (tiling**3, 3): coordinates on a cubic lattice,
        all between -0.5L and 0.5L
    """
    Ncube = tiling ** 3
    pos = np.zeros((Ncube, 3))
    temp = L / tiling
    t_offset = L / 2 - temp / 2
    t = 0
    for x in range(0, tiling):
        for y in range(0, tiling):
            for z in range(0, tiling):
                if t < Ncube:
                    pos[t, 0] = temp * x - t_offset
                    pos[t, 1] = temp * y - t_offset
                    pos[t, 2] = temp * z - t_offset
                t += 1
    return pos
    pass


def get_temperature(mass, velococities):
    """
    calculates the instantaneous temperature
    required for: initial_velococities()
    
    args:
        mass (float): mass of particles;
        it is assumed all particles have the same mass
        velococities (array): velococities of particles,
        assumed to have shape (N, 3)
    returns:
        float: temperature according to equipartition
    """
    v_sum = 0

    for i in velococities:
        v_sum = v_sum + i ** 2
    avg_v = v_sum / len(velococities)
    return mass * avg_v
    pass


def initial_velococities(N, m, T):
    """
    initialize velococities at a desired temperature
    required for: initialization

    args:
        N (int): number of particles
        m (float): mass of particles
        T (float): desired temperature
    returns:
        array: initial velococities, with shape (N, 3)
    """
    veloc = np.random.rand(N, 3)
    sV = np.sum(veloc) / N
    veloc = veloc - sV
    veloc = veloc * np.sqrt(T / get_temperature(m, veloc))
    return veloc
    pass


def displacement_table(coordinates, L):
    """
    required for: force(), advance()

    args:
        coordinates (array): coordinates of particles,
        assumed to have shape (N, 3)
        e.g. coordinates[3,0] should give the x component
        of particle 3
        L (float): side length of cubic box,
        must be known in order to compute minimum image
    returns:
        array: table of displacements r
        such that r[i,j] is the minimum image of
        coordinates[i] - coordinates[j]
    """

    r = np.zeros((len(coordinates), len(coordinates), 3))
    for i in range(0, len(coordinates)):
        for j in range(0, len(coordinates)):
            r[i, j] = minimum_image(coordinates[i] - coordinates[j], L)
    return r
    pass


def kinetic(m, v):
    """
    required for measurement

    args:
        m (float): mass of particles
        v (array): velococities of particles,
        assumed to be a 2D array of shape (N, 3)
    returns:
        float: total kinetic energy
    """
    kin = 0
    for i in v:
        kin = kin + .5 * m * i**2 
    return kin


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

def v_func(r):
    return 4 * np.power((1 / r), 6) * (np.power((1 / r), 6) - 1)


def potential(dist, rc):
    """
    required for measurement

    args:
        dist (array): distance table with shape (N, N)
        i.e. dist[i,j] is the distance
        between particle i and particle j
        in the minimum image convention
        note that the diagonal of dist can be zero
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, the pair potential between
        i and j will be 0
    returns:
        float: total potential energy
    """
    shift = v_func(rc)
    p_sum = 0
    size = len(dist)
    for j in range(0, size - 1):
        for i in range(j + 1, size):
            if dist[j, i] >= rc:
                continue
            p_sum += v_func(dist[j, i]) - shift
    return p_sum


def f_force(ros):
    r = np.sqrt(np.power(ros[0], 2) + np.power(ros[1], 2) + np.power(ros[2], 2))
    return (1 / np.power(r, 2)) * np.power(1 / r, 6) * (2 * np.power(1 / r, 6) - 1) * ros


def force(disp, dist, rc):
    """
    warning: this computes the force on ALL particles
    rather than just one, as in the PrairieLearn exercise
    also: instead of taking the particle coordinates and length of box,
    this takes the displacement and distance tables,
    since these are computed externally

    required for: advance()

    args:
        disp (array): displacement table,
        with shape (N, N, 3)
        dist (array): distance table, with shape (N, N)
        can be calculated from displacement table,
        but since there is a separate copy available
        it is just passed in here
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, particle i will feel no force
        from particle j
    returns:
        array: forces f on all particles, with shape (N, 3)
        i.e. f[3,0] gives the force on particle i
        in the x direction
    """
    f_arr = np.zeros((len(disp), 3))
    for i in range(0, len(disp)):
        f_sum = np.array([0, 0, 0])
        size = len(disp)
        for j in range(0, size):
            if j == i:
                continue
            if rc <= dist[i, j]:
                continue
            f_sum = f_sum + f_force(disp[i, j])
        f_arr[i] = f_sum
    return f_arr
    pass


# def advance(pos, veloc, mass, dt, disp, dist, rc, L):
#     """
#     advance system according to velococity verlet

#     args:
#         pos (array): coordinates of particles
#         val (array): velococities of particles
#         mass (float): mass of particles
#         dt (float): timestep by which to advance
#         disp (array): displacement table
#         dist (array): distance table
#         rc (float): cutoff
#         L (float): length of cubic box
#     returns:
#         array, array, array, array:
#         new positions, new velococities, new displacement table,
#         and new distance table
#     """
#     accel = force(disp, dist, rc) / mass
#     # move
#     veloc_half = veloc + 0.5 * dt * accel
#     pos_new = pos + dt * veloc_half
#     disp_new = displacement_table(pos_new, L)
#     dist_new = np.linalg.norm(disp_new, axis=-1)
#     # repeat force calculation for new pos
#     accel = force(disp_new, dist_new, rc) / mass
#     # finish move
#     veloc_new = veloc_half + 0.5 * dt * accel
#     return pos_new, veloc_new, disp_new, dist_new

def potential_energy(i, pos, L, rc):
  """
  Args:
    i (int): particle index
    pos (np.array): particle positions, shape (number of atoms, 3)
    L (float): cubic box side length
    rc (float): potential cutoff radius
  Return:
    float: potential energy for particle i
  """
  vshift = 4*rc**(-6)*(rc**(-6)-1)  # potential shift
  r = minimum_image(pos[i] - pos, L)
  # r[j] is the same as the minimum image of pos[i] - pos[j]
  # it has shape (number of atoms, 3)

  # complete this function
  pe = 0.0
  return potential_helper(r, rc)
  
def potential_helper(rij, rc):
    v =0.0 
    for j in range(len(rij)): 
        rr = math.sqrt(rij[j][0]**2 + rij[j][1]**2 +rij[j][2]**2)
        if rr < rc and rr != 0:
            v = v + 4*(1/rr)**6*((1/rr)**6-1) - 4*(1/rc)**6*((1/rc)**6-1)
    return v 

def log_trial_probability(fold, fnew, eta, tau, beta):
  """ Calculate the logarithm of the ``proposal'' contribution to
   acceptance probability in a smart Monte Carlo move.

  Args:
    fold (np.array): shape (ndim), force before move
    fnew (np.array): shape (ndim), force after move
    eta (np.array): shape (ndim), Gaussian random move vector
    sigma (float): width of Gaussian random numbers (eta0)
    beta (float): inverse temperature
  Return:
    float: ln(T(r'->r)/T(r->r'))
  """
  Tr = np.linalg.norm(eta)**2/4/tau
  TrP = np.linalg.norm(-eta-tau*beta*fold -tau*beta*fnew)** 2/4/tau
  return Tr - TrP 

def min_image(pos, lbox): 
    given_pos = np.copy(pos)
    for i in range(3):
        if given_pos[i] > 0 :
            given_pos[i] = given_pos[i] -lbox*np.ceil((given_pos[i]-lbox/2)/lbox)
        elif given_pos[i] < 0:
            given_pos[i] = given_pos[i] + lbox*np.ceil((-given_pos[i]-lbox/2)/lbox)
    return given_pos
    
def my_equation(r, rc):
    rr = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    if rr > rc:
        return 0
    else:
        return (2/(rr**6)-1)/(rr**8) * r

def my_force_on(i, pos, lbox, rc):
    """
    Compute force on atom i, cut off pair interaction beyond rc

    Args:
      i (int): particle index
      pos (np.array) : particle positions, shape (natom, ndim)
      lbox (float): side length of cubic box
      rc (float): cutoff radius
    Returns:
      np.array: force on atom i, a length-3 vector
    """
    temp = 0
    cutoff = rc
    length = len(pos)
    for x in range(length):
        if x == i:
            continue 

        temp += my_equation(min_image(pos[i]-pos[x],lbox), cutoff)
    return temp *24

# for reproducibility
np.random.seed(466)

# parameters
num_particles = 64
temperature = 0.728
length = 4.232317
cutoff = 2.08
mass = 48
timestep = 0.001

# system
coordinates = cubic_lattice(4, length)
velococities = initial_velococities(num_particles, mass, temperature)

# tables required to compute quantities like forces, energies
displacements = displacement_table(coordinates, length)
distances = np.linalg.norm(displacements, axis=-1)

# advance and record energies
# it can also be useful to save coordinates and velococities
KE = []
PE = []
a_rates = [] 
start_time = time.time()

for _ in range(10000):
    # coordinates, velococities, displacements, distances = advance(coordinates, \
    #                                                             velococities, mass, timestep, displacements, distances,
    #                                                             cutoff, \
    #                                                             length)

    rates = []
    old_force = force(displacements, distances, cutoff)
    # for i in range(0, len(coordinates)):
    #     initial_coordinates = coordinates[i]
    #     eta = []
    #     for j in range(0 ,3):
    #         eta.append(random.gauss(0, np.sqrt(2* timestep)))
    #     old_potential = potential_energy(i, coordinates, length,cutoff)
    #     coordinates[i] += eta
    #     new_potential = potential_energy(i, coordinates, length, cutoff)
    #     dv = np.abs(new_potential - old_potential)
    #     rate =np.minimum(1,np.exp(-1 * dv/temperature))
    #     rates.append(rate)
    #     if random.uniform(0,1) >= rate: 
    #         coordinates[i] = initial_coordinates
    a_rates.append(np.mean(rates))
    rates = []
    pot =[]
    fold = force(displacements, distances, cutoff)
    for i in range(0,len(coordinates)):
        potential_average =  np.mean(potential_energy(i, coordinates, length,cutoff))
        pot.append(potential_average)


    # old_pot = potential_energy(i, coordinates, length,cutoff)
    # initial_coord = coordinates[i]
    # eta = []
    # etas = np.random.normal(0,np.sqrt(2*timestep), (3,))
    # coordinates[i] += etas +timestep*1/temperature*fold[i]
    # fnew = my_force_on(i, coordinates, length, cutoff)
    # new_potential = potential_energy(i, coordinates, length, cutoff)
    # pot.append(new_potential)
    # dv = np.abs(new_potential- old_pot)    
    # rate = np.minimum(1,np.exp(-1*dv/temperature + log_trial_probability(fold[i], fnew, etas,timestep, 1/temperature)))
    # rates.append(rate)
    # if random.uniform(0,1) >= rate: 
    #     coordinates[i] = initial_coord
    print(np.var(pot))
    print(np.mean(my_actime(pot)))




        




# write to disk
np.savetxt('results.csv', KE)