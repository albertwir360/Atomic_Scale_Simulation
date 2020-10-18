import numpy as np


def minimum_image(r, L):
    """
    required for: displacement_table(), advance()
    args:
        r: array of any shape
        L: side length of cubic box
    returns:
        array of the same shape as r,
        the minimum image of r
    """
    given_pos = np.copy(r)
    for i in range(0, len(r)):
        if given_pos[i] > 0:
            given_pos[i] = given_pos[i] - L * np.ceil((given_pos[i] - L / 2) / L)
        elif given_pos[i] <= 0:
            given_pos[i] = given_pos[i] + L * np.ceil((-given_pos[i] - L / 2) / L)
    return given_pos
    pass


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
    print(veloc)
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


def advance(pos, veloc, mass, dt, disp, dist, rc, L):
    """
    advance system according to velococity verlet

    args:
        pos (array): coordinates of particles
        val (array): velococities of particles
        mass (float): mass of particles
        dt (float): timestep by which to advance
        disp (array): displacement table
        dist (array): distance table
        rc (float): cutoff
        L (float): length of cubic box
    returns:
        array, array, array, array:
        new positions, new velococities, new displacement table,
        and new distance table
    """
    accel = force(disp, dist, rc) / mass
    # move
    veloc_half = veloc + 0.5 * dt * accel
    pos_new = pos + dt * veloc_half
    disp_new = displacement_table(pos_new, L)
    dist_new = np.linalg.norm(disp_new, axis=-1)
    # repeat force calculation for new pos
    accel = force(disp_new, dist_new, rc) / mass
    # finish move
    veloc_new = veloc_half + 0.5 * dt * accel
    return pos_new, veloc_new, disp_new, dist_new


# for reproducibility
np.random.seed(466)

# parameters
num_particles = 64
temperature = 0.728
length = 4.232317
cutoff = 2.08
mass = 48
timestep = 0.01

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
for _ in range(2000):
    coordinates, velococities, displacements, distances = advance(coordinates, \
                                                                velococities, mass, timestep, displacements, distances,
                                                                cutoff, \
                                                                length)
    PE.append(potential(distances, cutoff))
    KE.append(kinetic(mass, velococities))

# write to disk
np.savetxt('results.csv', KE)