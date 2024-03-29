from simplex_class import Simplex
from problem_class import Problem
from scipy.optimize import linprog

def compare_results(problem: Problem, simplex: Simplex):
	status_dict = {	0: 'optimal',
					1: 'Iteration limit reached.',
					2: 'infeasible',
					3: 'unbounded',
					4: 'Serious numerical difficulties encountered.'}
		
	c = problem.c
	A = problem.A
	b = problem.b

	res = linprog(c=c, A_eq=A, b_eq=b, bounds=(0, None), method='highs')
	simplex.solve(problem)

	if problem.state == 'optimal' and status_dict[res.status] == 'optimal':
		if abs(res.fun - problem.Z) > simplex.tolerance:
			print(f'Problem({problem.data_id}, {problem.problem_id})')
			print(f'\t*SciPy: {res.fun}')
			print(f'\t*Our implementation: {problem.Z}')
			print(f'\t*Abs. Error: {abs(res.fun - problem.Z)}')
			print('---------------------------------')

	elif (problem.state != status_dict[res.status]) and (problem.state != 'infeasible (degeneracy)' and status_dict[res.status] != 'infeasible'):
		print(f'Problem({problem.data_id}, {problem.problem_id})')
		print(f'\t*SciPy: {status_dict[res.status]}')
		print(f'\t*Our implementation: {problem.state}')
		print('---------------------------------')

if __name__ == '__main__':
	data_ids = range(1, 67)
	problem_ids = range(1, 5)
	problems = []
	for d in data_ids:
		for problem in problem_ids:
			problem = Problem(data_id=d, problem_id=problem)
			problems.append(problem)

	simplex = Simplex(print_results=False, print_iters=False, save_results=False, tolerance=1e-10)
	
	for problem in problems:
		compare_results(problem, simplex)