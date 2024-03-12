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
		self.__phase2()
		pass

	##### PRIVATE METHODS ----------------------------------------------------------------------------------------------- #

	#### Methods for each phase

	def __initialize_values(self, problem: Problem, is_artificial: bool = True) -> None:
		self.m, self.n = problem.A.shape


		pass

	def __phase1(self) -> None:
		self.__generate_artificial_problem()
		self.__run(problem=self.artificial_problem)

		pass

	def __phase2(self) -> None:
		self.__run(problem=self.problem)
		pass

	### Methods for the execution of the algorithm

	def __run(self, problem: Problem) -> None:
		self.__initialize_values(problem=problem)
		pass

	## Methods for each step

	def __step1(self) -> None:
		pass

	def __step2(self) -> None:
		pass

	def __step3(self) -> None:
		pass

	def __step4(self) -> None:
		pass

	def __step5(self) -> None:
		pass

	# Other methods

	def __generate_artificial_problem(self) -> None:
		"""
		Generates the artificial problem for the phase 1 of the simplex algorithm.
		"""
		A =
		c =
		b =
		
		self.artificial_problem = Problem(A=A, c=c, b=b)
	def __update_B_inv(self) -> None:
		pass

	def __reduced_costs(self):
		pass

	def __calculate_solution(self) -> None:
		pass

	def __select_enetering_variable(self):
		pass

	def __select_leaving_variable(self):
		pass

	def __pivot(self, entering, leaving) -> None:
		pass

	def __is_optimal(self) -> None:
		pass
