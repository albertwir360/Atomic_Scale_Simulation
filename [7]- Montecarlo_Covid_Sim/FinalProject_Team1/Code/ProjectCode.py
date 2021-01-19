import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import matplotlib.lines as mlines

def minimum_image(r, L):
	# Use minimum image in range [0,L] , not [-L/2,L/2]
	r_shifted = r - 0.5 * L
	return r_shifted - L * np.round( r_shifted / L ) + 0.5 * L

def GenerateMove( mean , SD ):
	# Direction [-pi,pi]
	direction = np.random.rand() * 2 * np.pi - np.pi
	# Distance drawn from a normal dist
	distance = np.random.normal( mean , SD )
	return np.array([ np.cos(direction)*distance , np.sin(direction) ])



class Agent():
	def __init__(self):
		# Available states:
		# 0: Suspectible
		# 1: Exposed
		# 2: Infected
		# 3: Quarantined
		# 4: Dead
		self.state = 0

		# # Position of the agent
		# self.position = np.zeros(2)

		# Mask?
		self.mask = False

		# Number of steps exposed before getting infected
		self.exposedDays = 0
		self.quarantineDays = 0


class Simulation():
	def __init__(self):
		# Size of UIUC main quad
		# self.size = np.array([ 983. , 390. ])

		# Random box
		self.size = np.array([ 150. , 150. ])


		# All population in the domain
		self.population = []
		self.positions = []

		# Dummy parameters
		self.N_total_agents = 0
		self.N_sim_steps = 0
		self.N_initial_infected = 0
		self.pct_with_mask = 0
		self.Social_distancing = False
		self.Socially_awkward = False
		self.Have_gatherings = False

		# Walking statistics
		self.mean_walk_dist = 25 # ft
		self.SD_walk_dist = 10 # ft

		# KD tree
		self.CurrentTree = None

		# State lists
		self.infectedList = None
		self.partyList = None
		self.partyLocations = None
		self.party_positions = []


		#probablity of quarantine
		# self.prob_quarantine = 0.0

		#probability of death
		self.prob_death = 0.003

		#number of deaths total
		self.totalDeaths = 0

		#number of cases total
		self.totalCases = 0

		# probability of recovering
		self.prob_recover = 0.2

		# probability coefficients
		# P = pre_factor * mask_coeff * ( 1 - tanh( d / d_coeff) )
		self.pre_factor = 0.7
		self.mask_coeff = 0.5
		# Larger d_coeff means longer tail, more likely to get infected
		self.d_coeff = 3

		# Latent period
		self.latent_period = 14

		# How many parties to throw 
		self.N_parties = 1

		# Radius of influence of the party
		self.partyRadius = 15. # ft


	def SetParameters( self , a , b , c , d , e , f , g  ):
		self.N_total_agents = a
		self.N_sim_steps = b
		self.N_initial_infected = c
		self.pct_with_mask = d
		self.Social_distancing = e
		self.Have_gatherings = f
		self.prob_quarantine = g


	def PlotState( self ):
		s0 = mlines.Line2D([], [], color='g', marker='o', markersize=12, label='Healthy')
		s1 = mlines.Line2D([], [], color='m', marker='d', markersize=12, label='Exposed')
		s2 = mlines.Line2D([], [], color='r', marker='P', markersize=12, label='Infected')
		s3 = mlines.Line2D([], [], color='g', marker='P', markersize=12, label='Infected but quarantined')
		m = mlines.Line2D([], [], color='b', marker='1', markersize=12, label='Wears mask')
		p = mlines.Line2D([], [], color='y', marker='s', markersize=12, label='Gathering holder')

		# Plot outer contour
		plt.plot( [0,self.size[0]] , [0,0] , 'k' )
		plt.plot( [0,self.size[0]] , [self.size[1],self.size[1]] , 'k' )
		plt.plot( [0,0] , [0,self.size[1]] , 'k' )
		plt.plot( [self.size[0],self.size[0]] , [0,self.size[1]] , 'k' )


		StateCount = np.zeros(5)
		for idx in range(len(self.population)) :
			ag = self.population[idx]
			code = ''
			if ag.state == 0 :
				code = 'go'
				StateCount[0] += 1
			elif ag.state == 1 :
				code = 'md'
				StateCount[1] += 1
			elif ag.state == 2 :
				code = 'rP'
				StateCount[2] += 1
			elif ag.state == 3 :
				code = 'gP'
				StateCount[3] += 1
			else:
				StateCount[4] += 1

			if ag.state != 4:
				plt.plot( self.positions[idx][0] , self.positions[idx][1] , code )

				if ag.mask :
					plt.plot( self.positions[idx][0] , self.positions[idx][1] , 'b1' )

		if self.Have_gatherings:
			for _ in self.partyList:
				plt.plot( self.positions[_][0] , self.positions[_][1] , 'ys' )

		ax = plt.gca()
		ax.axis('equal')
		ax.set_xlim( 0 , self.size[0] )
		ax.set_ylim( 0 , self.size[1] )
		plt.legend(handles=[s0,s1,s2,s3,m,p])

		StateCount /= ( float(len(self.population)) / 100. )
		print('\n>>> State statistics:')
		print('	Healthy: ' + str(StateCount[0]) + '%' )
		print('	Exposed: ' + str(StateCount[1]) + '%' )
		print('	Infected (not quarantined): ' + str(StateCount[2]) + '%' )
		print('	Quarantined: ' + str(StateCount[3]) + '%' )
		print('	Removed: ' + str(StateCount[4]) + '%' )




	def initialize( self , N_initial_steps ):
		# Step 1: Set the initial positions
		Number = int(np.ceil(np.sqrt(self.N_total_agents))) + 2

		# All available coords
		x = np.linspace( 0.05 * self.size[0] , 0.95 * self.size[0], Number )
		y = np.linspace( 0.05 * self.size[1] , 0.95 * self.size[1], Number )

		idx = list(range( Number * Number ))
		np.random.shuffle( idx )

		for i in range( self.N_total_agents ):
			# Get the 2D index
			row = idx[i] // Number
			col = idx[i] - row * Number

			ag = Agent()
			self.population.append( ag )
			self.positions.append( np.array([ x[col] , y[row] ]) )


		# Step 2: Choose the initial infectors (exposed)
		self.infectedList = np.random.choice( self.N_total_agents , self.N_initial_infected )
		for i in self.infectedList:
			self.population[i].state = 1
		self.infectedList = set(self.infectedList)


		# Step 3: Choose the mask wearers
		N_mask = int(np.ceil( self.N_total_agents * self.pct_with_mask / 100. ))
		for i in np.random.choice( self.N_total_agents , N_mask ):
			self.population[i].mask = True

		
		# Step 4: Choose the party holders
		self.partyList = set( np.random.choice( self.N_total_agents , self.N_parties ) )


		# Step 5: Warm up the system without any transmission
		avg_rate , _ = self.Move( N_initial_steps, False )
		print( 'Initial warm-up:\n	Avg accptance rate of {} percent'.format( avg_rate ) )

		# Step 6: Store party locations
		if self.Have_gatherings:
			for _ in self.partyList:
				self.party_positions.append( self.positions[_] )
			self.partyLocations = KDTree( np.array(self.party_positions) , leaf_size=2 )		


	def Move( self , N_steps, doStateUpdate ):
		Accptance = []
		StateHistory = np.zeros([ 5 , N_steps + 1 ])
		StateHistory[0][:] = self.N_total_agents - self.N_initial_infected
		StateHistory[1][:] = self.N_initial_infected

		for i in range(N_steps):
			StateHistory[:,i+1] = StateHistory[:,i]

			# Like a MC, let's keep track of the accptance ratio
			N_accpeted = 0

			# Pass through all agents
			for a in range(self.N_total_agents):
				# When we study the infection dynamics, fix the party location and let agents join the party
				if ( doStateUpdate and a in self.partyList ):
					continue

				# Ignore the dead and the quarantined 	
				if self.population[a].state == 4 or self.population[a].state == 3 : 
					continue

				# Store old position
				my_old_pos = np.copy( self.positions[a] )

				# Make a temporary move
				if self.Have_gatherings and doStateUpdate:
					# How far am I to the nearest party?
					dis , idx = self.partyLocations.query( [self.positions[a]] , k=1 )
					min_dist_to_party = dis[0][0]

					# Determine whether to join the party or not
					if min_dist_to_party <= 20:
						self.positions[a] = self.party_positions[ idx[0][0] ] + GenerateMove( 5 , 0.01 )
					else:
						self.positions[a] += GenerateMove( self.mean_walk_dist , self.SD_walk_dist )

				else:
					self.positions[a] += GenerateMove( self.mean_walk_dist , self.SD_walk_dist )
				
				# Enforce image convention
				self.positions[a] = minimum_image( self.positions[a] , self.size )

				# Update the curent KD tree
				self.CurrentTree = KDTree( np.array(self.positions) , leaf_size=50 )

				# Find the nearest neighbour
				dis , IDX = self.CurrentTree.query( [self.positions[a]] , k=2 )
				minDis = dis[0][-1] - 1.5 # Avg. width of adult shoulder

				# Decide if the move is acceptable
				if self.Social_distancing :
					minAcceptableDis = 6.
				else:
					minAcceptableDis = 0.

				if minDis > minAcceptableDis :
					# 2 ft is the radius of a intimate zone 
					A = np.exp( - minDis / 2. )
				else:
					A = 1.

				if np.random.rand() > A:
					# We accept the move
					N_accpeted += 1

					# Update agent state
					Me = self.population[a]
					Me_position = self.positions[a]

					################################################################## TO DO  #################################################################################
					if doStateUpdate:
						if self.population[a].state == 0:
							# How far am I to the nearest infected/exposed agent?
							Inf_Exp_positions = []
							for _ in self.infectedList:
								Inf_Exp_positions.append( self.positions[_] )
							if Inf_Exp_positions == []:
								min_dist_to_infected = np.inf
							else:
								Inf_Exp_tree = KDTree( np.array(Inf_Exp_positions) , leaf_size=50 )

								# Find the closest distance to the infected from the position of agent
								dis , idx = Inf_Exp_tree.query( [Me_position] , k=1 )
								min_dist_to_infected = dis[0][0]
							
							# Will I get sick?
							# Am I wearing a mask?
							if Me.mask:
								probability = self.pre_factor * self.mask_coeff * ( 1. - np.tanh( min_dist_to_infected / self.d_coeff ) )
							else:
								probability = self.pre_factor * ( 1. - np.tanh( min_dist_to_infected / self.d_coeff ) )
							
							# print( min_dist_to_infected , probability )


							# Do a random draw to determine state
							if np.random.sample() <= probability:
								self.population[a].state = 1
								StateHistory[0][i+1] -= 1
								StateHistory[1][i+1] += 1
								self.totalCases += 1

						elif self.population[a].state == 1:
							# Keep track of changes
							if a not in self.infectedList :
								# This means that we have a newly exposed member
								# append index to list of infected
								self.infectedList.add( a )

							# Increment # of exposed days
							self.population[a].exposedDays += 1

							# Switch to infected after a certain number of days
							if self.population[a].exposedDays == self.latent_period: 
								# Change state to infected
								self.population[a].state = 2 
								StateHistory[1][i+1] -= 1
								StateHistory[2][i+1] += 1
						else:
							out_of_list = False

							# Am I going into quarantine?
							# Change 2 to 3 using random draw
							if np.random.sample() <= self.prob_quarantine and self.population[a].state == 2: 
								self.population[a].state = 3
								out_of_list = True
								StateHistory[2][i+1] -= 1
								StateHistory[3][i+1] += 1

							# Increment quarantine days until you recover
							if self.population[a].state == 3:
								self.population[a].quarantineDays += 1

							# If not, am I going to die?
							if np.random.sample() <= self.prob_death:
								# If dead, remove from population 
								self.totalDeaths += 1 
								self.population[a].state = 4
								out_of_list = True	
								StateHistory[3][i+1] -= 1
								StateHistory[4][i+1] += 1

							# Am I going to recover?
							elif np.random.sample() <=  self.prob_recover:
								self.population[a].state = 0
								self.population[a].exposedDays = 0
								out_of_list = True
								StateHistory[3][i+1] -= 1
								StateHistory[0][i+1] += 1

							# Book-keeping
							if out_of_list:
								# Remove from tracking list
								try:
									self.infectedList.remove( a )
								except:
									pass

					################################################################## TO DO  #################################################################################



				else:
					# We reject the move, restore old position
					self.positions[a] = my_old_pos

			Accptance.append( N_accpeted / self.N_total_agents * 100. )
		return np.mean( Accptance ) , StateHistory




###############################################################################################
# User inputs
# Global setups
N_total_agents = 80
N_sim_steps = 1

# Initial conditions
N_initial_infected = 10
Quarantine_adherence = 0.96 #See [1] BBC UK poll
Test_accuracy = 1 #See [2] UIUC Saliva Test
Quarantine_prob = Test_accuracy*Quarantine_adherence
Social_distancing = False
Have_gatherings = False


###############################################################################################

# Task 1: Effect of mask wearing
# for pct_with_mask in [ 0. , 25. , 50. , 75. , 100. ]:
for pct_with_mask in [96.0]:
	Sim_name = 'SD'+ str( Social_distancing ) + '-HG' + str( Have_gatherings ) + '-m' + str( int(pct_with_mask) ) + '-q' + str(Quarantine_prob)
	# Repeat simulation for stats
	Data = []
	IFRData = []
	PNCData = []
	for inst in range( N_sim_steps ):
		# Build a new simulation
		mySim = Simulation()
		mySim.SetParameters( N_total_agents , N_sim_steps , N_initial_infected , pct_with_mask , Social_distancing , Have_gatherings , Quarantine_prob )
		# Initialize agent moves without transmission
		mySim.initialize( 10 )
		# Run with transmission dynamics
		ap , StateHistory = mySim.Move( 250 , True )
		Data.append( StateHistory )

		# plt.figure()
		# plt.title('After transmission period '+ Sim_name)
		# mySim.PlotState()
		#
		# Plot state history
		# plt.figure()
		# plt.title( Sim_name )
		# plt.plot( StateHistory[0] , 'g' )
		# plt.plot( StateHistory[1] , 'm' )
		# plt.plot( StateHistory[2] , 'r' )
		# plt.plot( StateHistory[3] , 'b' )
		# plt.plot( StateHistory[4] , 'k' )
		# plt.legend(['Suspectible','Exposed','Infected','Quarantined','Removed'])
		# plt.show()

		#Print Infection Fatality Rate (IFR) for Calibration
		IFR = 0
		if mySim.totalCases > 0:
			IFR = round(mySim.totalDeaths/mySim.totalCases*100,2)
		IFRData.append(IFR)
		# print('IFR is ' + str(IFR) + '%')

		#Print Population Normalized Cases (PNC) for Calibration
		PNC = round(mySim.totalCases/N_total_agents*100,2)
		PNCData.append(PNC)
		#print('PNC is ' + str(PNC) +'%'

		print(str(inst+1)+' of '+str(N_sim_steps))

	# Write to disk
	f = open( Sim_name + '.npy' , 'wb' )
	np.save(f,Data)
	f.close()

	f = open( 'IFR_'+Sim_name + '.npy' , 'wb' )
	np.save(f,IFRData)
	f.close()

	f = open( 'PNC_'+Sim_name + '.npy' , 'wb' )
	np.save(f,PNCData)
	f.close()


print('Run Complete')