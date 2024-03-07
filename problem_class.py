from typing import Optional
from numpy.typing import NDArray
import numpy as np

class Problem:
	def __init__(self, data_id: Optional[int] = None, problem_id: Optional[int] = None) -> None:
		assert (data_id is not None and problem_id is not None) or (data_id is None and problem_id is None), 'Both data_id and problem_id must be None or not None'

		self.data_id: Optional[int] = data_id
		self.problem_id: Optional[int] = problem_id

		self.c: Optional[NDArray[np.int64]] = None
		self.A: Optional[NDArray[np.int64]] = None
		self.b: Optional[NDArray[np.int64]] = None

		if data_id is not None and problem_id is not None:
			self.__read_problem()

	def __read_problem(self) -> None:
		with open(f'./data/data.txt', 'r') as file:
			line = file.readline()

			while f'datos {self.data_id}' not in line and f'PL {self.problem_id}' not in line:
				line = file.readline()

			while 'c=' not in line:
				line = file.readline()

			c_line = file.readline().strip()

			while 'A=' not in line:
				line = file.readline()

			A_lines = []
			line = file.readline().strip()
			while line:  
				A_lines.append(line)
				line = file.readline().strip()

			while 'b=' not in line:
				line = file.readline()

			b_line = file.readline().strip()

		self.c = np.array([int(x) for x in c_line.split()])
		self.A = np.array([[int(x) for x in line.split()] for line in A_lines if line])  # Asegura que line no esté vacío
		self.b = np.array([int(x) for x in b_line.split()])
			

			
			



					

					


