from simplex_class import Simplex
from problem_class import Problem
import numpy as np

if __name__ == '__main__':
	problems_ids = [(10, 1), (10, 2), (10, 3), (10, 4), (48, 1), (48, 2), (48, 3), (48, 4)]
	problems = [Problem(data_id=data_id, problem_id=problem_id) for data_id, problem_id in problems_ids]
	
	c = np.array([-1, 0, 0])

	b = np.array([4, 2])

	A = np.array([[1, 1, 1], [2, -1, 0]])

	p = Problem(A=A, c=c, b=b)

	s = Simplex()
	s.solve(p)




