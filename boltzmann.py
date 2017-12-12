"""
Written by Kevin Caleb Eades
Astro 207
Final Project
Fall 2017
"""

"""
Notes:
-- Einstein A values came from 
https://www.physics.byu.edu/faculty/christensen/Physics%20612/FTI/H%20Einstein%20Coefficients.htm
-- the multiplication by 100 of steps in the plot saving section is to prevent
	*.png from going in weird orders like 1, 10 rather than 1,2
-- I wrote the simulation as if the set of particles is in thermal equilibrium
	then using the spontaneous emission/absorption and stimulated emission to
	govern the transitions between states. I also cut it off after n=4 so there
	could be small discrepancies with the expected boltzmann distribution
	arising from that since higher n-values are not considered so transitions
	between them and n=1 to n=4 states presentned here are ignored.
-- All atoms start in the ground state
-- if you initialize s = simulation(), then you can check whether the
	probabilities are reasonable for the simulation with s.probs; similary, you
	can check whether the expected Boltzmann distribution to compare it to will
	be useful/non-trivial with s.expected
"""

# importing constants from a class file I wrote
import Constants

# imports from standard modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random



def Jbar(nu_0,temp):
	"""
	:nu_0: the frequency of a transition we're looking at
	:temp: the temperature

	:returns: the blackbody spectrum evaluated at nu_0
	"""
	c = Constants.constants()
	num = 2*c.h*nu_0**3
	denom = c.c**2*(np.exp(c.h*nu_0/(c.k*temp))-1)
	return num/denom



def Factorial(k):
	"""
	:k: integer

	:returns: the factorial of k
	"""
	if k==0:
		return 1
	return k*Factorial(k-1)



def Poisson(rate,k=0):
	"""
	:rate: (float) the poisson mean rate over some interval of observation
	:k: (int>=0) the number of events for the interval of observation

	:returns: the probability of seeing k events over the interval of
			  observation with rate events
	"""
	return rate**k*np.exp(-rate)/Factorial(k)



class simulation(object):
	""" class for the simulation """
	def __init__(self,num_atoms=50000,temperature=2e4,dt=1e-10):
		"""
		constructor

		:num_atoms: (int) the number of atoms to use in the simulation
		:temperature: (float) the temperature (note for some low temperatures
					  you can get overflows in the Jbar calculation)
		:dt: (float) the timestep
		"""
		self.constants = Constants.constants()
		self.num_atoms = num_atoms
		self.temp = temperature
		self.dt = dt

		self.A_21 = {(2,1):{(1,0):6.3e8}\
			,(3,1):{(1,0):1.7e8,(2,0):2.2e7}\
			,(4,1):{(1,0):6.8e7,(2,0):9.7e6,(3,0):3.1e6,(3,2):3.5e5}\
			,(3,0):{(2,1):6.3e6}\
			,(3,2):{(2,0):2.2e7}\
			,(4,0):{(2,1):2.6e6,(3,1):1.8e6}\
			,(4,2):{(2,1):2.1e7,(3,1):7.0e6}\
			,(4,3):{(3,2):1.4e7}} # for spontaneous emission

		E_0 = -13.6*self.constants.eV
		self.energies = {i:E_0/i**2 for i in range(1,5)}
		self.freqs = {i:{j:0 for j in range(i+1,5)} for i in range(1,4)}
		for i in self.freqs:
			for j in self.freqs[i]:
				self.freqs[i][j] = (self.energies[j]-self.energies[i])\
					/self.constants.h

		self.B_21 = {} # for stimulated emission
		for start in self.A_21:
			self.B_21[start] = {}
			for end in self.A_21[start]:
				n_low = end[0]
				n_high = start[0]
				B = self.A_21[start][end]*self.constants.c**2/(2*self.constants\
					.h*self.freqs[n_low][n_high]**3)
				self.B_21[start][end] = B*Jbar(self.freqs[n_low][n_high]\
					,self.temp)
		
		self.B_12 = {} # for absorption
		for start in self.B_21:
			for end in self.B_21[start]:
				if end not in self.B_12:
					self.B_12[end] = {}
				n_low = end[0]
				n_high = start[0]
				self.B_12[end][start] = (2*n_high+1)/(2*n_low+1)\
					*self.B_21[start][end]

		self.probs = {} # dictionary of probabilities to transition from one
						# state to another in the timescale dt
		for start in self.A_21:
			self.probs[start] = {}
			for end in self.A_21[start]:
				rate = (self.A_21[start][end] + self.B_21[start][end])*self.dt
				self.probs[start][end] = 1-Poisson(rate)
		for start in self.B_12:
			if start not in self.probs:
				self.probs[start] = {}
			for end in self.B_12[start]:
				rate = self.B_12[start][end]*self.dt
				self.probs[start][end] = 1-Poisson(rate)

		self.expected = [1] + [(2*i+1)/3*np.exp(-self.constants.h*self.freqs[1]\
			[i]/(self.constants.k*self.temp)) for i in range(2,5)]
		self.expected = np.divide(self.expected,np.sum(self.expected))

		self.Initialize()

	def Initialize(self):
		""" initializes the atoms """
		self.atoms = [atom() for i in range(self.num_atoms)]
		self.states = [0 for i in range(1,5)]
		self.states[0] = self.num_atoms

	def UpdateAtom(self,atom,update_mode='ordered'):
		"""
		updates a single atom to a potentially new state, quasi-randomly

		:atom: an atom object
		:update_mode: 'random' or 'ordered' -- random goes through final
						events randomly to see if they are triggered whereas
						ordered sorts by which one is most probable and walks
						down the list. ordered can be problematic if the
						probabilities of individual transitions are high as they
						will continually be triggered without the others getting
						a chance; meanwhile random can be problematic for the
						same reason if all individual transitions are high
		"""
		current = (atom.n,atom.l)
		possible = [(final,self.probs[current][final]) for final \
			in self.probs[current]]
		if update_mode=='random':
			random.shuffle(possible)
		if update_mode=='ordered':
			possible.sort(key=lambda x:x[1])
		final = current
		for state in possible:
			if np.random.random()<state[1]:
				final = state[0]
				break
		atom.Update(final[0],final[1])
		self.states[current[0]-1] -= 1
		self.states[final[0]-1] += 1

	def Iterate(self):
		""" runs through one timestep and updates the atoms """
		for atom in self.atoms:
			self.UpdateAtom(atom)

	def Run(self,steps=100,saving=True,save_name='animation'):
		"""
		runs the simulationereasereasereas

		:steps: (int) the total number of timesteps to use
		:saving: (bool) whether to save the result to a gif or not
		:save_name: (str) the file name for the gif to be saved to
		"""
		if saving:
			self.files = []
		self.Compare(0,steps,saving)
		for step in range(steps):
			self.Iterate()
			self.Compare(step+1,steps,saving)
		if saving:
			os.system("convert -delay 15 -loop 0 temp_*.png " + save_name \
				+ '.gif')
			for fname in self.files:
				os.remove(fname)

	def Compare(self,step,steps,saving=True):
		"""
		creates a comparison plot of the current atomic state distribution
		compared to what we expect

		:step: (int) the current step number in the simulation
		:steps: (int) the total number of steps to use
		:saving: (bool) whether to save the results or not
		"""
		plt.ioff()
		plt.figure()
		plt.plot(np.arange(1,5),self.expected,'o-',color='black',label\
			='expected')
		plt.plot(np.arange(1,5),np.divide(self.states,np.sum(self.states))\
			,'o-',color='green',label='actual')
		plt.legend()
		plt.title('Expected vs Actual Atom Distribution at State ' + str(step))
		plt.xlabel('n (principle quantum number)')
		plt.ylabel('Fraction of Atoms')
		plt.ylim(0,1)
		plt.xlim(1,4)
		if saving:
			self.files.append('temp_' + str(step+steps*100)+'.png')
			plt.savefig('temp_' + str(step+steps*100) + '.png')
		plt.close()



class atom(object):
	"""
	atom class, unneccesary at the moment but maybe useful if/when adding
	more features to this simulation
	"""
	def __init__(self):
		""" constructor """
		self.n = 1
		self.l = 0

	def Update(self,n,l):
		"""
		updates the atom's atomic numbers

		:n: (int) principle quantum number
		:l: (int) orbital angular momentum quantum number
		"""
		self.n = n
		self.l = l



if __name__=='__main__':
	"""
	example of how to run the simulation and change around parameters
	"""
	num_atoms = 25000
	temp = 5e5
	dt = 1e-11
	s = simulation(num_atoms,temp,dt)

	steps = 200
	saving = True
	name = 'testing' + '_animation'
	s.Run(steps,saving,name)