import numpy as np

class constants(object):
	"""constants class for use in Astro 207 (these are some of the
	more commonly used ones to date). Here is a description of what
	is accessible here.

	constants Object Class
	____________________________________________________
	The following is available with the appropriate call
	(note that all units are CGS):
	Constants::
	-- speed of light -> .c
	-- electron mass -> .me
	-- Boltzmann constant -> .k
	-- Stefan-Boltzmann constantn -> .sigma
	-- reduced Planck constant -> .hbar
	-- electron charge -> .e
	-- gravitational acceleration -> .g
	-- gravitational constant -> .G
	-- mass of a nucleon -> .mn
	-- classical electron radius -> .re
	-- Bohr radius -> .a0
	-- fine structure constant -> .alpha
	-- Einstein A coefficient of Ly-alpha -> .A
	-- Planck constant -> .h
	-- Thomson cross section -> .sigmaT
	Conversions::
	-- eV to ergs -> .eV
	-- parsec to cm -> self.parsec
	-- Newtons to dynes -> .newton
	-- Debye to cgs -> .debye
	Astronomical masses and distances::
	-- radius of the earth -> .rad_e
	-- radius of the sun -> .rad_s
	-- distance to the moon -> .dist_m
	-- distance to the sun -> .dist_s
	-- distance to galactic center -> .dist_g
	-- mass of the earth -> .mass_e
	-- mass of the sun -> .mass_s
	Other astronomical things::
	-- luminosity of the sun -> .ls
	-- critical density -> .rho_cr
	-- baryon density -> .rho_b"""

	def __init__(self):
		""" constructor """
		# constants
		self.c = 3e10
		self.me = 1e-27
		self.k = 1.4e-16
		self.sigma = 6e-5
		self.hbar = 1e-27
		self.e = 5e-10
		self.g = 1e3
		self.G = 7e-8
		self.mn = 2000*self.me
		self.re = 3e-13
		self.a0 = 5e-9
		self.alpha = 1/137
		self.A = 5e8
		self.h = 2*np.pi*self.hbar
		self.sigmaT = 7e-25

		# conversions
		self.eV = 1.6e-12
		self.parsec = 3e18
		self.newton = 1e5
		self.debye = 1e-18

		# astronomical masses/distances
		self.rad_e = 6e8
		self.rad_s = 8e10
		self.dist_m = 4e10
		self.dist_s = 1.5e13
		self.dist_g = 7e3*self.parsec
		self.mass_e = 6e27
		self.mass_s = 2e33

		# other astronomical things
		self.ls = 4e33
		self.rho_cr = 2e-29*0.72**2
		self.rho_b = 0.02*self.rho_cr

	def __str__(self):
		return """Constants Object: for help on what is available and
		how to access them, run 'help(x)' in your terminal where x is
		a constants object or constants() itself."""

	def __repr__(self):
		return """Constants Object: for help on what is available and
		how to access them, run 'help(x)' in your terminal where x is
		a constants object or constants() itself."""