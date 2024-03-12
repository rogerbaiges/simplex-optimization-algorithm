from typing import Optional
from problem_class import Problem
from numpy.typing import NDArray
import numpy as np

class Simplex:
	def __init__(self) -> None:
		self.problem: Optional[Problem] = None
		self.artificial_problem: Optional[Problem] = None
		self.B_variables: Optional[NDArray] = None
		self.N_variables: Optional[NDArray] = None
		self.B: Optional[NDArray] = None
		self.B_inv: Optional[NDArray] = None
		self.C_B: Optional[NDArray] = None
		self.C_N: Optional[NDArray] = None
		self.X_B: Optional[NDArray] = None
		self.Z: Optional[np.float64] = None
		self.n: Optional[np.int32] = None
		self.m: Optional[np.int32] = None

	##### PUBLIC METHODS ------------------------------------------------------------------------------------------------ #

	def solve(self, problem: Problem) -> None:
		assert problem.c is not None, 'The problem must have the c vector'
		assert problem.A is not None, 'The problem must have the A matrix'
		assert problem.b is not None, 'The problem must have the b vector'
		self.problem = problem

		self.__phase1()
		if self.Z > 0:
			print('The problem is infeasible.')
		else:
			self.__phase2()
			print(f'The problem was solved and the solutions are:\nvb*={self.problem.solution}\nZ={self.Z})')

	##### PRIVATE METHODS ----------------------------------------------------------------------------------------------- #

	#### Methods for each phase

	def __initialize_values(self, problem: Problem, is_artificial: bool = True) -> None:
		self.m, self.n = problem.A.shape

		if is_artificial:
			self.B_variables = np.arange(start=self.n - self.m, stop=self.n, step=1, dtype=np.int32)
			self.N_variables = np.arange(start=0, stop=self.n - self.m, step=1, dtype=np.int32)
			self.B = np.identity(n=self.m, dtype=np.int32)
			self.B_inv = np.identity(n=self.m, dtype=np.int32)
			self.C_B = np.ones(self.m, dtype=np.int32)
			self.C_N = np.zeros(self.n - self.m, dtype=np.int32)
			self.X_B = problem.b
			self.__calculate_z()
		else:
			assert self.B_variables is not None, 'The B_variables must be initialized'
			assert self.B_inv is not None, 'The B_inv must be initialized'
			# B y B_inv las tenemos
			# N_variables (buscar las variables que no aparecen en B_variables)
			self.N_variables = np.setdiff1d(np.arange(start=0, stop=self.n, step=1, dtype=np.int32), self.B_variables)
			self.C_B = problem.c[self.B_variables]
			self.C_N = problem.c[self.N_variables]
			self.X_B = np.dot(self.B_inv, problem.b)
			self.__calculate_z()

		print(f'INITIAL VALUES:\n\nB_variables={self.B_variables}\n\nN_variables={self.N_variables}\n\nB={self.B}\n\nB_inv={self.B_inv}\n\nC_B={self.C_B}\n\nC_N={self.C_N}\n\nX_B={self.X_B}\n\nZ={self.Z}\n')

	def __phase1(self) -> None:
		self.__generate_artificial_problem()
		self.__run(problem=self.artificial_problem)
		return self.Z
		

	def __phase2(self) -> None:
		self.__run(problem=self.problem)

	### Methods for the execution of the algorithm

	def __run(self, problem: Problem) -> None:
		self.__initialize_values(problem=problem, is_artificial=False if problem == self.problem else True)
		
		# Execute all the simplex algorithm until an optimal solution is found or the problem is unbounded

		i = 0
		while True:
			print('Iteration', i)
			entering = self.__select_entering_variable()
			if entering is None: # The problem is optimal
				self.__calculate_solution()
				print('The problem is optimal')
				break
			
			leaving = self.__select_leaving_variable(entering=entering)
			if leaving is None: # The problem is unbounded
				print('The problem is unbounded')
				break
			
			self.__pivot(entering, leaving)

			i += 1
		
	# Other methods

	def __generate_artificial_problem(self) -> None:
		"""
		Generates the artificial problem for the phase 1 of the simplex algorithm.
		"""
		n, m = self.problem.A.shape
		A = np.concatenate((self.problem.A, np.identity(n=n)), axis=1)
		c = np.concatenate((np.zeros(n), np.ones(m)))
		b = self.problem.b
		self.artificial_problem = Problem(A=A, c=c, b=b)

	def __update_B_inv(self) -> None:
		"""
		Updates the B_inv matrix of the problem.
		"""
		self.B_inv = np.linalg.inv(self.B)
		return self.B_inv

	def __calculate_reduced_costs(self):
		"""
		Calculates the reduced costs of the problem.
		"""
		reduced_costs = self.C_N - np.dot(np.dot(self.C_B, self.B_inv), self.problem.A)
		return reduced_costs
	
	def __calculate_feasible_basic_direction(self, entering: int):
		"""
		Calculates the feasible basic direction of the problem.
		"""
		d_B = -np.dot(self.B_inv, self.problem.A[:, entering])
		return d_B

	def __calculate_solution(self):
		"""
		Calculates the values of all the variables of the problem.
		"""
		self.problem.solution = np.zeros(self.n)
		self.problem.solution[self.B_variables] = self.X_B
		return self.problem.solution

	def __select_entering_variable(self):
		"""
		Returns the entering variable of the problem using the Bland's rule.
		"""
		reduced_costs = self.__calculate_reduced_costs()

		if np.all(reduced_costs >= 0):
			return None
		
		for j, cost in zip(self.N_variables, reduced_costs):
			if cost < 0:
				return j

	def __select_leaving_variable(self, entering: int):
		"""
		Returns the leaving variable of the problem using the Bland's rule.
		"""
		d_B = self.__calculate_feasible_basic_direction(entering=entering)

		if np.all(d_B >= 0):
			return None
		
		theta = np.inf
		for i, d in enumerate(d_B):
			if d < 0:
				theta_i = -self.X_B[i] / d
				if theta_i < theta:
					theta = theta_i
					leaving = i
		return leaving

	def __pivot(self, entering, leaving) -> None:
		"""
		Executes the pivot operation of the simplex algorithm.
		"""
		self.B[:, leaving] = self.problem.A[:, entering]
		self.B_inv = self.__update_B_inv()
		self.B_variables[leaving] = entering
		self.N_variables = np.setdiff1d(self.N_variables, entering)
		self.C_B[leaving] = self.problem.c[entering]
		self.C_N = self.problem.c[self.N_variables]
		self.X_B = self.X_B - (self.X_B[leaving] / self.B_inv[leaving, leaving]) * self.B_inv[:, leaving]
		self.__calculate_z()

		print(f'FINAL ITER VALUES:\n\nB_variables={self.B_variables}\n\nN_variables={self.N_variables}\n\nB={self.B}\n\nB_inv={self.B_inv}\n\nC_B={self.C_B}\n\nC_N={self.C_N}\n\nX_B={self.X_B}\n\nZ={self.Z}\n')


	def __calculate_z(self):
		self.Z = np.dot(self.C_B, self.X_B)
		return self.Z