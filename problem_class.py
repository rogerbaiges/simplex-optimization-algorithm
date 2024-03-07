from typing import Optional
import numpy as np

class Problem:
	def __init__(self, data_id: Optional[int] = None, problem_id: Optional[int] = None) -> None:
		assert (data_id is not None and problem_id is not None) or (data_id is None and problem_id is None), 'Both data_id and problem_id must be None or not None'

		self.data_id: Optional[int] = None
		self.problem_id: Optional[int] = None

		self.c: Optional[list[int]] = None
		self.A: Optional[list[list[int]]] = None
		self.b: Optional[list[int]] = None
		self.z_opt: Optional[float] = None
		self.vb_opt: Optional[list[int]] = None

		if data_id is not None and problem_id is not None:
			self.read_problem()

	def read_problem(self) -> None:
		with open(f'../data/data.txt', 'r') as file:
			pass