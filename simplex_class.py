from typing import Optional
from problem_class import Problem
import numpy as np

class Simplex:
	def __init__(self, problem: Problem) -> None:
			self.problem: Problem = problem
			self.artificial_problem: Optional[Problem] = None

	##### PUBLIC METHODS ------------------------------------------------------------------------------------------------ #

	def solve(self) -> None:
		self.__phase1()
		self.__phase2()
		pass

	##### PRIVATE METHODS ----------------------------------------------------------------------------------------------- #

	#### Methods for each phase

	def __phase1(self) -> None:
		self.artificial_problem = Problem()
		self.__run(self.artificial_problem)

		pass

	def __phase2(self, ) -> None:
		pass

	### Methods for the execution of the algorithm

	def __run(self, problem: Problem) -> None:
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

	def __update_B_inv(self) -> None:
		pass





















	# def __init__(self, c, A, b):
	# 	self.c = c
	# 	self.A = A
	# 	self.b = b
	# 	self.m = len(A)
	# 	self.n = len(A[0])
	# 	self.B = list(range(self.n - self.m, self.n))
	# 	self.N = list(range(self.n - self.m))
	# 	self.x = [0] * self.n
	# 	self.z = 0

	# def pivot(self, l, e):
	# 	self.B[l], self.N[e] = self.N[e], self.B[l]
	# 	self.A[l] = [self.A[l][j] / self.A[l][e] for j in range(self.n)]
	# 	self.b[l] /= self.A[l][e]
	# 	for i in range(self.m):
	# 		if i != l:
	# 			self.A[i] = [self.A[i][j] - self.A[i][e] * self.A[l][j] for j in range(self.n)]
	# 			self.b[i] -= self.A[i][e] * self.b[l]
	# 	self.z += self.c[self.N[e]] * self.b[l]
	# 	self.c = [self.c[j] - self.c[self.N[e]] * self.A[l][j] for j in range(self.n)]
	# 	self.x[self.N[e]] = self.b[l]
	# 	self.x = [0 if j not in self.B else self.x[j] for j in range(self.n)]

	# def solve(self):
	# 	while min(self.c) < 0:
	# 		e = self.c.index(min(self.c))
	# 		if all(self.A[i][e] <= 0 for i in range(self.m)):
	# 			return None
	# 		l = min(i for i in range(self.m) if self.A[i][e] > 0 and self.b[i] / self.A[i][e] >= 0)
	# 		self.pivot(l, e)
	# 	return self.x