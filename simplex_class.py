from typing import Optional, Union, Literal, Tuple, TextIO
from problem_class import Problem
from numpy.typing import NDArray
import numpy as np

class Simplex:
	def __init__(self, tolerance: float = 1e-10, print_results: bool = False, print_iters: bool = False, save_results: bool = True) -> None:
		self.problem: Optional[Problem] = None
		self.artificial_problem: Optional[Problem] = None

		self.B_variables: Optional[NDArray] = None
		self.N_variables: Optional[NDArray] = None
		self.A_N: Optional[NDArray] = None
		self.B_inv: Optional[NDArray] = None
		self.C_B: Optional[NDArray] = None
		self.C_N: Optional[NDArray] = None
		self.X_B: Optional[NDArray] = None
		self.Z: Optional[np.float64] = None
		self.n: Optional[np.int32] = None
		self.m: Optional[np.int32] = None

		self.tolerance: float = tolerance

		self.print_results: bool = print_results
		self.print_iters: bool = print_iters
		self.save_results: bool = save_results

		self.results_file: Optional[TextIO] = None

	#### PUBLIC METHODS ------------------------------------------------------------------------------------------------ #

	def solve(self, problem: Problem) -> None:
		"""
		Solves the given problem using the simplex algorithm.
		"""
		assert problem.c is not None, 'The problem must have the c vector'
		assert problem.A is not None, 'The problem must have the A matrix'
		assert problem.b is not None, 'The problem must have the b vector'
		self.problem = problem

		if self.save_results and (self.problem.data_id is None) and (self.problem.problem_id is None):
			self.save_results = False
			print('Warning: The results will not be saved because the problem has not been initialized with data_id and problem_id.')

		self.__phase1()
		if self.print_results:
			print(f'\n-------------------[Problem: {self.problem.data_id}, {self.problem.problem_id}]-------------------\n')

		if self.save_results:
			self.__write_problem_results_in_results_file(problem=self.artificial_problem)
		
		if self.print_results:
			self.__print_problem_results(problem=self.artificial_problem)	

		self.__check_if_feasible()		   
		
		if self.problem.state is None: # If the problem is not infeasible
			self.__phase2()
			
			if self.save_results:
				self.__write_problem_results_in_results_file(problem=self.problem)
			
			if self.print_results:
				self.__print_problem_results(problem=self.problem)
		
		else: # If the problem is infeasible
			if self.print_results:
				self.__print_problem_results(problem=self.problem)
			
			if self.save_results:
				self.results_file.write('\nPHASE II: Not executed because the problem is infeasible.\n')
		
		if self.save_results:
			self.__write_final_solution_in_results_file()

	#### PRIVATE METHODS ----------------------------------------------------------------------------------------------- #

	### Methods for each phase

	def __initialize_values(self, problem: Problem, is_artificial: bool = False) -> None:
		"""
		Initializes the values of the problem for the simplex algorithm.
		"""
		self.m, self.n = problem.A.shape

		if is_artificial:
			self.B_variables = np.arange(start=self.n - self.m, stop=self.n, step=1, dtype=np.int32)
			self.N_variables = np.arange(start=0, stop=self.n - self.m, step=1, dtype=np.int32)
			self.B_inv = np.identity(n=self.m, dtype=np.int32)
			self.A_N = problem.A[:, self.N_variables]
			self.C_B = np.ones(self.m, dtype=np.int32)
			self.C_N = np.zeros(self.n - self.m, dtype=np.int32)
			self.X_B = problem.b
			self.__calculate_z()
			if self.save_results:
				self.__initialize_results_file()
				self.results_file.write('PHASE I:\n')
		else:
			assert self.B_variables is not None, 'The B_variables must be initialized'
			assert self.B_inv is not None, 'The B_inv must be initialized'
			# B_variables and B_inv are already calculated in the artificial problem
			self.N_variables = np.setdiff1d(np.arange(start=0, stop=self.n, step=1, dtype=np.int32), self.B_variables)
			self.A_N = problem.A[:, self.N_variables]
			self.C_B = problem.c[self.B_variables]
			self.C_N = problem.c[self.N_variables]
			self.X_B = np.dot(self.B_inv, problem.b)
			self.__calculate_z()
			if self.save_results:
				self.results_file.write('\nPHASE II:\n')

		if self.print_iters:
			self.__print_initial_values(problem=problem)

	def __phase1(self) -> Tuple[str, int]:
		"""
		Runs the phase 1 of the simplex algorithm.
		"""
		self.__generate_artificial_problem()
		finish_state, iters = self.__run(problem=self.artificial_problem, is_artificial=True)
		self.artificial_problem.state = finish_state
		self.artificial_problem.iterations = iters
		return finish_state, iters

	def __phase2(self) -> Tuple[str, int]:
		"""
		Runs the phase 2 of the simplex algorithm.
		"""
		finish_state, iters = self.__run(problem=self.problem, is_artificial=False)
		self.problem.state = finish_state
		self.problem.iterations = iters
		return finish_state, iters

	## Methods for the execution of the algorithm

	def __run(self, problem: Problem, is_artificial: bool = False) -> Tuple[str, int]:
		"""
		Runs the simplex algorithm for the given problem.
		"""
		self.__initialize_values(problem=problem, is_artificial=is_artificial)
		
		state: Optional[Union[Literal['optimal'], Literal['unbounded'], Literal['infeasible']]] = None
		iter = 1
		
		while True:
			reduced_costs = self.__calculate_reduced_costs()
			if self.__is_optimal(reduced_costs=reduced_costs):
				self.__save_solution_in_problem(problem=problem, reduced_costs=reduced_costs)
				state = 'optimal'
				break
			index_entering = self.__select_entering_variable(reduced_costs=reduced_costs)

			d_B = self.__calculate_basic_feasible_direction(index_entering=index_entering)
			if self.__is_unbounded(d_B=d_B):
				state = 'unbounded'
				break
			index_leaving, theta = self.__select_leaving_variable(d_B=d_B)

			if self.print_iters:
				word_print = 'Art. ' if is_artificial else ''
				print(f'---------------[{word_print}Problem: {self.problem.data_id}, {self.problem.problem_id} | Iter. {iter}]---------------')
			
			var_entering = self.N_variables[index_entering]
			var_leaving = self.B_variables[index_leaving]

			self.__pivot(problem=problem, index_entering=index_entering, index_leaving=index_leaving, var_entering=var_entering, var_leaving=var_leaving, theta=theta, d_B=d_B, reduced_costs=reduced_costs)
			self.__calculate_z(theta=theta, r_q=reduced_costs[index_entering])

			if self.save_results:
				self.__write_iteration_results(iter=iter, var_entering=var_entering, var_leaving=var_leaving, index_entering=index_entering, index_leaving=index_leaving, theta=theta)

			iter += 1

		return state, iter - 1
				
	# Auxiliary methods

	def __generate_artificial_problem(self) -> None:
		"""
		Generates the artificial problem for the phase 1 of the simplex algorithm.
		"""
		m, n = self.problem.A.shape
		A = np.concatenate((self.problem.A, np.identity(n=m)), axis=1)
		c = np.concatenate((np.zeros(n), np.ones(m)))
		b = self.problem.b
		self.artificial_problem = Problem(A=A, c=c, b=b)

	def __update_B_inv(self, index_leaving: int, d_B: NDArray) -> NDArray:
		"""
		Updates the B_inv matrix of the problem.
		"""
		E = np.identity(self.m)
		E[:, index_leaving] = -d_B / d_B[index_leaving]
		E[index_leaving, index_leaving] = -1 / d_B[index_leaving]
		self.B_inv = np.dot(E, self.B_inv)
		return self.B_inv

	def __calculate_reduced_costs(self):
		"""
		Calculates the reduced costs of the problem.
		"""
		return self.C_N - np.dot(np.dot(self.C_B, self.B_inv), self.A_N)
	
	def __calculate_basic_feasible_direction(self, index_entering: int) -> NDArray:
		"""
		Calculates the basic feasible direction of the problem.
		"""
		d_B = -np.dot(self.B_inv, self.A_N[:, index_entering])
		return d_B

	def __save_solution_in_problem(self, problem: Problem, reduced_costs: NDArray) -> None:
		"""
		Calculates the values of all the variables of the problem.
		"""
		problem.xb = self.X_B
		problem.Z = self.Z
		problem.vb = self.B_variables
		problem.r = reduced_costs

	def __select_entering_variable(self, reduced_costs: NDArray) -> Optional[int]:
		"""
		Returns the index of the entering variable of the problem using the Bland's rule.
		"""
		for i in np.argsort(self.N_variables):
			if reduced_costs[i] < 0:
				return i

	def __select_leaving_variable(self, d_B: NDArray) -> Optional[Tuple[int, np.float64]]:
		"""
		Returns the index of the leaving variable of the problem using the Bland's rule, the theta value and the feasible basic direction.
		"""
		theta = np.inf
		for i in np.argsort(self.B_variables):
			if d_B[i] < 0:
				theta_i = -self.X_B[i] / d_B[i]
				if theta_i < theta:
					theta = theta_i
					index_leaving = i
		return index_leaving, theta

	def __pivot(self, problem: Problem, index_entering: int, index_leaving: int, var_entering: int, var_leaving: int, theta, d_B: NDArray, reduced_costs: NDArray) -> None:
		"""
		Executes the pivot operation of the simplex algorithm (updates the values of the problem after an iteration).
		"""	
		self.B_variables[index_leaving] = var_entering
		self.N_variables[index_entering] = var_leaving
		self.A_N[:, index_entering] = problem.A[:, var_leaving]
		self.B_inv = self.__update_B_inv(index_leaving=index_leaving, d_B=d_B)
		self.C_B[index_leaving] = problem.c[var_entering]
		self.C_N[index_entering] = problem.c[var_leaving]
		self.X_B = np.array([theta if i == index_leaving else self.X_B[i] + theta * d_B[i] for i in range(self.m)])

		if self.print_iters:
			self.__print_final_iter_values(reduced_costs=reduced_costs, d_B=d_B, theta=theta, var_entering=var_entering, var_leaving=var_leaving)

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
	
	def __check_if_feasible(self) -> None:
		"""
		Checks if the problem is feasible. If it is not, it changes the state of the problem to 'infeasible'.
		"""
		if self.artificial_problem.Z > self.tolerance:
			self.problem.state = 'infeasible'

	def __calculate_z(self, theta = None, r_q = None) -> np.float64:
		"""
		Calculates, updates and returns the current Z value of the problem.
		"""
		if (theta is not None) and (r_q is not None):
			assert self.Z is not None, 'The Z value must be initialized before updating it'
			self.Z = self.Z + theta * r_q
		else:
			self.Z = np.dot(self.C_B, self.X_B)
		return self.Z

	# Methods for printing and saving (writing to .txt) the results

	def __initialize_results_file(self) -> None:
		"""
		Initializes the results file.
		"""
		open(f'./results/results_d{self.problem.data_id}_p{self.problem.problem_id}.txt', 'w') # Clear the file
		self.results_file = open(f'./results/results_d{self.problem.data_id}_p{self.problem.problem_id}.txt', 'a') # Open the file in append mode

	def __write_iteration_results(self, iter: int, var_entering: int, var_leaving: int, index_entering: int, index_leaving: int, theta: np.float64) -> None:
		"""
		Saves the results of the iteration in the results file.
		"""
		self.results_file.write(f'\t* ITER. {iter} --> Var. IN = {var_entering} (N_variables[{index_entering}]), Var. OUT = {var_leaving} (B_variables[{index_leaving}]), Theta* = {theta:.4f}, Z = {self.Z:.4f}\n')

	def __write_final_solution_in_results_file(self) -> None:
		"""
		Writes the final solution of the problem in the results file.
		"""
		self.results_file.write(f'\n----------------------------------------\n\n')
		if self.problem.state == 'optimal':
			self.results_file.write(f'The problem is optimal. The solution found is:\n\tvb = {self.problem.vb}\n\txb = {self.problem.xb}\n\tr = {self.problem.r}\n\tZ = {self.problem.Z}\n')
		elif self.problem.state == 'unbounded':
			self.results_file.write('The problem is unbounded.\n')
		else:
			self.results_file.write('The problem is infeasible.\n')

	def __write_problem_results_in_results_file(self, problem: Problem) -> None:
		"""
		Writes the results of the problem in the results file.
		"""
		word_print = 'artificial ' if problem is self.artificial_problem else ''
		self.results_file.write(f'\t* Finished {word_print}problem after {problem.iterations} iterations with state: {problem.state.capitalize()}.\n')
		self.results_file.write(f'\t* Solution:\n\t\tvb = {problem.vb}\n\t\txb = {problem.xb}\n\t\tr = {problem.r}\n\t\tZ = {problem.Z}\n')

	def __print_problem_results(self, problem: Problem) -> None:
		"""
		Prints the results of the problem.
		"""
		is_artificial = problem is self.artificial_problem
		print('\n-------------------\n') if not is_artificial else None
		word_print = 'artificial ' if is_artificial else ''
		if problem.state == 'infeasible':
			print(f'The problem is infeasible.')
		else:
			print(f'Finished {word_print}problem after {problem.iterations} iterations with state: {problem.state.capitalize()}.')
			print(f'Solution:\n\tvb = {problem.vb}\n\txb = {problem.xb}\n\tr = {problem.r}\n\tZ = {problem.Z}')

	def __print_initial_values(self, problem: Problem) -> None:
		"""
		Prints the initial values of the problem.
		"""
		word_print = 'ART. ' if problem is self.artificial_problem else ''
		print(f'--------------- INITIAL VALUES {word_print}PROBLEM ---------------\n\nB_variables={self.B_variables}\n\nN_variables={self.N_variables}\n\nB_inv=\n{self.B_inv}\n\nA_N=\n{self.A_N}\n\nC_B={self.C_B}\n\nC_N={self.C_N}\n\nX_B={self.X_B}\n\nZ={self.Z}\n')
	
	def __print_final_iter_values(self, reduced_costs: NDArray, d_B: NDArray, theta: np.float64, var_entering: int, var_leaving: int) -> None:
		"""
		Prints the final values of the iteration.
		"""
		print(f'FINAL ITER. VALUES:\n\nR={reduced_costs}\n\nd_B={d_B}\nTheta={theta}\n\nVar. IN --> {var_entering}\nVar. OUT --> {var_leaving}\n\nB_variables={self.B_variables}\nN_variables={self.N_variables}\n\nB_inv=\n{self.B_inv}\n\nA_N=\n{self.A_N}\n\nX_B={self.X_B}\n\nC_B={self.C_B}\nC_N={self.C_N}\n\nZ={self.Z}\n')
