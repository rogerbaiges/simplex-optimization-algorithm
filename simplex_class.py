from typing import Optional, Union, Literal, Tuple
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

		ph1_finish_state = self.__phase1()
		print(f'Finished artificial problem ({ph1_finish_state})')
		
		if ph1_finish_state == 'optimal':
			ph2_finish_state = self.__phase2()
			print(f'Finished problem ({ph2_finish_state})')
			print(f'Solutions:\nvb* = {self.problem.solution}\nZ = {self.Z}')

	##### PRIVATE METHODS ----------------------------------------------------------------------------------------------- #

	#### Methods for each phase

	def __initialize_values(self, problem: Problem, is_artificial: bool = False) -> None:
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
			# B_variables, B and B_inv are already calculated in the artificial problem
			self.N_variables = np.setdiff1d(np.arange(start=0, stop=self.n, step=1, dtype=np.int32), self.B_variables)
			self.C_B = problem.c[self.B_variables]
			self.C_N = problem.c[self.N_variables]
			self.X_B = np.dot(self.B_inv, problem.b)
			self.__calculate_z()

		word_print = 'ART. ' if is_artificial else ''
		print(f'--------------- INITIAL VALUES {word_print}PROBLEM ---------------\n\nB_variables={self.B_variables}\n\nN_variables={self.N_variables}\n\nB=\n{self.B}\n\nB_inv=\n{self.B_inv}\n\nC_B={self.C_B}\n\nC_N={self.C_N}\n\nX_B={self.X_B}\n\nZ={self.Z}\n')

	def __phase1(self) -> str:
		self.__generate_artificial_problem()
		finish_state = self.__run(problem=self.artificial_problem, is_artificial=True)
		return finish_state

	def __phase2(self) -> str:
		finish_state = self.__run(problem=self.problem, is_artificial=False)
		return finish_state

	### Methods for the execution of the algorithm

	def __run(self, problem: Problem, is_artificial: bool = False) -> str:
		self.__initialize_values(problem=problem, is_artificial=is_artificial)
		
		state: Optional[Union[Literal['optimal'], Literal['unbounded'], Literal['infeasible']]] = None
		iter = 1
		
		while True:
			reduced_costs = self.__calculate_reduced_costs()
			if self.__is_optimal(reduced_costs=reduced_costs):
				self.__calculate_solution()
				state = 'optimal'
				break
			index_entering = self.__select_entering_variable(reduced_costs=reduced_costs)

			d_B = self.__calculate_feasible_basic_direction(index_entering=index_entering)
			if self.__is_unbounded(d_B=d_B):
				state = 'unbounded'
				break
			index_leaving, theta = self.__select_leaving_variable(d_B=d_B)

			word_print = 'Art. ' if is_artificial else ''
			print(f'---------------[{word_print}Problem: Iter. {iter}]---------------')
			
			self.__pivot(problem=problem, index_entering=index_entering, index_leaving=index_leaving, theta=theta, d_B=d_B, reduced_costs=reduced_costs)

			iter += 1

		if is_artificial:
			if self.Z != 0:
				print(f'The problem is infeasible with Z = {self.Z}')
				state = 'infeasible'

		return state
				
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

	def __update_B_inv(self) -> NDArray:
		"""
		Updates the B_inv matrix of the problem.
		"""
		self.B_inv = np.linalg.inv(self.B)
		return self.B_inv

	def __calculate_reduced_costs(self):
		"""
		Calculates the reduced costs of the problem.
		"""
		return self.C_N - np.dot(np.dot(self.C_B, self.B_inv), self.problem.A)
	
	def __calculate_feasible_basic_direction(self, index_entering: int) -> NDArray:
		"""
		Calculates the feasible basic direction of the problem.
		"""
		d_B = -np.dot(self.B_inv, self.problem.A[:, index_entering])
		return d_B

	def __calculate_solution(self):
		"""
		Calculates the values of all the variables of the problem.
		"""
		self.problem.solution = np.zeros(self.n)
		self.problem.solution[self.B_variables] = self.X_B
		return self.problem.solution

	def __select_entering_variable(self, reduced_costs: NDArray) -> Optional[int]:
		"""
		Returns the index of the entering variable of the problem using the Bland's rule.
		"""
		for i, r_q in enumerate(reduced_costs):
			if r_q < 0:
				return i

	def __select_leaving_variable(self, d_B: NDArray) -> Optional[Tuple[int, np.float64]]:
		"""
		Returns the index of the leaving variable of the problem using the Bland's rule, the theta value and the feasible basic direction.
		"""	
		theta = np.inf
		for i, d in enumerate(d_B):
			if d < 0:
				theta_i = -self.X_B[i] / d
				if theta_i < theta:
					theta = theta_i
					index_leaving = i
		return index_leaving, theta

	def __pivot(self, problem: Problem, index_entering: int, index_leaving: int, theta, d_B: NDArray, reduced_costs: NDArray) -> None:
		"""
		Executes the pivot operation of the simplex algorithm.
		"""
		var_entering = self.N_variables[index_entering]
		var_leaving = self.B_variables[index_leaving]
		
		self.B[:, index_leaving] = problem.A[:, index_entering]
		self.B_inv = self.__update_B_inv()
		self.B_variables[index_leaving] = var_entering
		self.N_variables[index_entering] = var_leaving
		self.C_B[index_leaving] = problem.c[var_entering]
		self.C_N[index_entering] = problem.c[var_leaving]
		self.X_B = self.X_B + theta * d_B
		self.__calculate_z(theta, reduced_costs[index_entering])

		print(f'FINAL ITER VALUES:\n\nd_B={d_B}\nTheta={theta}\n\nVariable entering --> {var_entering}\nVariable leaving --> {var_leaving}\n\nB_variables={self.B_variables}\nN_variables={self.N_variables}\n\nB=\n{self.B}\n\nB_inv=\n{self.B_inv}\n\nX_B={self.X_B}\n\nC_B={self.C_B}\nC_N={self.C_N}\n\nZ={self.Z}\n')
	
	def __is_optimal(self, reduced_costs: NDArray) -> bool:
		"""
		Returns True if the problem is optimal.
		"""
		return np.all(reduced_costs >= 0)
	
	def __is_unbounded(self, d_B: NDArray) -> bool:
		"""
		Returns True if the problem is unbounded.
		"""
		return np.all(d_B >= 0)

	def __calculate_z(self, theta = None, r_q = None) -> np.float64:
		if (theta is not None) and (r_q is not None):
			assert self.Z is not None, 'The Z value must be initialized before updating it'
			self.Z = self.Z + theta * r_q
		else:
			self.Z = np.dot(self.C_B, self.X_B)
		return self.Z