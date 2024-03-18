from simplex_class import Simplex
from problem_class import Problem
import numpy as np

if __name__ == '__main__':
	problems_ids = [(10, 1), (10, 2), (10, 3), (10, 4), 
				 	(48, 1), (48, 2), (48, 3), (48, 4)]
	problems = [Problem(data_id=data_id, problem_id=problem_id) for data_id, problem_id in problems_ids]
	simplex = Simplex(print_results=False, print_iters=False, save_results=True)

	for p in problems:
		simplex.solve(p)