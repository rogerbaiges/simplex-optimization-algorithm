from typing import Optional
from numpy.typing import NDArray
import numpy as np
import re

class Problem:
	def __init__(self, 
			  	data_id: Optional[int] = None, 
			  	problem_id: Optional[int] = None, 
			  	dtype = np.int64,
			  	c: Optional[NDArray] = None,
				A: Optional[NDArray] = None,
				b: Optional[NDArray] = None
				) -> None:

		assert (data_id is None) == (problem_id is None), 'Both data_id and problem_id must be None or not None'
		assert (c is None) == (A is None) == (b is None), 'c, A and b must be all None or all not None'
		if (data_id is not None) and (c is not None):
			print('Warning: c, A, b are not None, but data_id, problem_id are not None too. The problem will be read from file.')

		self.dtype = dtype

		self.data_id: Optional[int] = data_id
		self.problem_id: Optional[int] = problem_id

		self.c: Optional[NDArray[dtype]] = c
		self.A: Optional[NDArray[dtype]] = A
		self.b: Optional[NDArray[dtype]] = b

		self.vb: Optional[NDArray] = None
		self.xb: Optional[NDArray] = None
		self.Z: Optional[np.float64] = None
		self.r: Optional[NDArray] = None
		self.state: Optional[str] = None
		self.iterations: Optional[int] = None

		if (data_id is not None) and (problem_id is not None):
			self.__read_problem()

	def __read_problem(self) -> None:
		with open(f'./data/data.txt', 'r') as file:
			truncated_input = False

			# Advancing to the required problem			
			regex_data_id = r'datos\s+{}'.format(re.escape(str(self.data_id)))
			regex_problem_id = r'PL\s+{}'.format(re.escape(str(self.problem_id)))
			line = file.readline()
			while True:
				if re.search(regex_data_id, line) and re.search(regex_problem_id, line):
					break
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

				if not 'b=' in line:
					for i in range(len(A_lines)):
						A_lines[i] += line.split()
						line = file.readline().strip()

			# Reading b
			while not 'b=' in line:
				line = file.readline()

			b_line = file.readline().strip().split()

		# Convert to numpy array
		self.c = np.array([self.dtype(x) for x in c_line])
		self.A = np.array([[self.dtype(x) for x in line] for line in A_lines])  # Asegura que line no esté vacío
		self.b = np.array([self.dtype(x) for x in b_line])

	def __str__(self) -> str:
		return f'DATA_ID={self.data_id}\nPROBLEM_ID={self.problem_id}\n\nSOLUTION ({self.state}):\n\tvb={self.vb}\n\n\txb={self.xb}\n\n\tZ={self.Z}\n\n\tr={self.r}'