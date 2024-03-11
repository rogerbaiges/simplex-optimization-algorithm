from typing import Optional
from numpy.typing import NDArray
import numpy as np

class Problem:
	def __init__(self, data_id: Optional[int] = None, problem_id: Optional[int] = None, dtype = np.int32) -> None:
		assert (data_id is not None and problem_id is not None) or (data_id is None and problem_id is None), 'Both data_id and problem_id must be None or not None'

		self.dtype = dtype

		self.data_id: Optional[int] = data_id
		self.problem_id: Optional[int] = problem_id

		self.c: Optional[NDArray[dtype]] = None
		self.A: Optional[NDArray[dtype]] = None
		self.b: Optional[NDArray[dtype]] = None

		self.solution: Optional[NDArray[dtype]]

		if data_id is not None and problem_id is not None:
			self.__read_problem()

	def __read_problem(self) -> None:
		with open(f'./data/data.txt', 'r') as file:
			truncated_input = False

			# Advancing to the required problem
			line = file.readline()
			while f'datos {self.data_id}' not in line and f'PL {self.problem_id}' not in line:
				line = file.readline()

			# Reading c
			while 'c=' not in line:
				line = file.readline()
			
			line = file.readline()
			while (not line.strip()) or ('column' in line.lower()):
				if 'column' in line.lower():
					truncated_input = True
				line = file.readline()

			c_line = line.strip().split()

			if truncated_input:
				line = file.readline()
				while (not line.strip()) or ('column' in line.lower()):
					line = file.readline()

				c_line += line.strip().split()

			# Reading A
			line = file.readline()
			while 'A=' not in line:
				line = file.readline()

			A_lines = []
			line = file.readline()
			while (not line.strip()) or ('column' in line.lower()):
				line = file.readline()

			line = line.strip()
			while line:  
				A_lines.append(line.split())
				line = file.readline().strip()

			if truncated_input:
				while (not line.strip()) or ('column' in line.lower()):
					line = file.readline()

				for i in range(len(A_lines)):
					A_lines[i] += line.split()
					line = file.readline().strip()

			# Reading b
			while 'b=' not in line:
				line = file.readline()

			b_line = file.readline().strip().split()

		# Convert to numpy array
		self.c = np.array([self.dtype(x) for x in c_line])
		self.A = np.array([[self.dtype(x) for x in line] for line in A_lines])  # Asegura que line no esté vacío
		self.b = np.array([self.dtype(x) for x in b_line])

	def __str__(self) -> str:
		return f'c=\n{self.c}\n\nA=\n{self.A}\n\nb=\n{self.b}\n\nSolution=\n{self.solution}'