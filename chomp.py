import numpy as np
import sys
from gym.envs.toy_text import discrete
import pdb
import copy
import random

H = 4
W = 7

# ACTIONS = [i for i in range(0, H*W)]
# print(ACTIONS)

def get_all_states():
	S = []
	for h in range(W + 1):
		# print(h)
		for i in range(h + 1):
			# print(f'\t{i}')
			for j in range(i + 1):
				# print(f'\t\t{j}')
				for k in range(0, j+1):
					S.append((h, i, j, k))
					# print((h, i, j, k))
	return S

class ChompEnv(discrete.DiscreteEnv):
	"""
	0 0 0 0 0 0 0
	0 0 0 0 0 0 0
	0 0 0 0 0 0 0
	0 0 0 0 0 0 0
	Top-right tile is the 0th tile.
	Bottom-left tile is the 27th tile.
	"""
	def __init__(self):
		n = W + 1
		nS = int(0.5 * (-1 * (n * (n+1) / 2) ** 2 + (n**2) * (n + 1) * (2 * n + 1) / 6 + n * ((n+1) ** 2) / 2))
		# print(nS)
		nA = H*W

		self.states = get_all_states()
		assert len(self.states) == nS

		P = {}
		for i, s in enumerate(self.states):

			# P[s] = {a : [] for a in range(nA)}
			P[s] = {}

			# is_done = lambda s: s == (W, W, W, W)

			if self.is_done(s):
				# opponents losses
				# for a in range(nA):
				# 	P[s][a] = [1.0, s, 0.0, self.is_done(s)]
				break
			else:
				for a in range(nA):
					if self.is_action_valid(a, s):
						next_s = self.get_next_state(a, s)
						reward = -10.0 if self.is_done(next_s) else 0.0
						P[s][a] = [1.0, next_s, reward, self.is_done(next_s)]
					# else:
						# P[s][a] = None


		isd = np.ones(nS) / nS
		self.P = P

		super(ChompEnv, self).__init__(nS, nA, P, isd)

	def is_done(self, s):
		return s == (W, W, W, W)

	def get_valid_actions(self, s):
		actions = []
		for a in range(self.nA):
			if self.is_action_valid(a, s):
				actions.append(a)
		return actions # e.g. [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27]

	def is_action_valid(self, a, s):
		row, col = self.action_index_to_row_col(a)
		if col > s[row]: # state[row] = 0, 1, 2, 3, 4, 5, 6 or 7
			return True
		else:
			return False 

	def get_next_state(self, a, s):
		row, col = self.action_index_to_row_col(a)
		next_s = []
		for i in range(H):
			if i > row:
				next_s.append(s[i])
				continue
			# print(f'{row} {col} {s[i]}')
			if col > s[i]:
				next_s.append(col)
			else:
				next_s.append(s[i])
		return tuple(next_s)

	def action_index_to_row_col(self, a):
		row = a // W # 0, 1, 2 or 3
		col = a % W + 1 # 1, 2, 3, 4, 5, 6 or 7
		return (row, col)

	def row_col_to_action_index(self, row, col):
		i = row * H + col - 1
		return i

	def select_random_actions(self, s):
		return random.choice(env.get_valid_actions(s))

	def get_next_state_upon_random_action(self, s):
		random_a = self.select_random_actions(s)
		# print(f'random_a: {random_a}')
		return self.get_next_state(random_a, s), random_a

	def get_next_state_info_upon_action(self, a, s):
		next_s = self.get_next_state(a, s)
		done = self.is_done(next_s)
		reward = -10.0 if done else 0.0
		return 1.0, next_s, reward, done

	def get_next_state_info_following_policy(self, policy, s):
		best_a = max(policy[s], key=policy[s].get)
		next_s = self.get_next_state(best_a, s)
		done = self.is_done(next_s)
		reward = -10.0 if done else 0.0
		return 1.0, next_s, reward, done

	def get_best_actions(self, policy):
		result = {}
		for s in self.P:
			if policy[s]:
				best_a = max(policy[s], key=policy[s].get)
				result[s] = best_a
		return result

env = ChompEnv()
gamma = 1.0
# generated_s = set()
# generated_a = set()
# for i in range(1000):
# 	s, a = env.get_next_state_upon_random_action((3,3,3,0))
# 	generated_a.add(a)
# 	generated_s.add(s)
# print(generated_a)
# print(generated_s)

# print(env.get_next_state(11, (3,3,3,0)))
# print(env.get_valid_actions((6, 5, 3, 0)))

def policy_iteration():
	policy = {}
	count = 0
	for s, v in env.P.items():
		# print(f'{s} {v}')
		s_policy = {}
		if v:
			prob_a = 1 / len(v)
			for a in v:
				s_policy[a] = prob_a
				count += 1
		# s_policy = np.ones(len(v)) / len(v)
		policy[s] = s_policy
		# for a in v:
		# 	print(f'\t{a}')
	# print(policy)

	# print(count)

	epoch = 100
	for i in range(epoch):
		print(f'Epoch: {i+1}')
		V = policy_evaluation(policy)
		# if i % 20 == 0:
		# 	pdb.set_trace()
		old_policy = copy.deepcopy(policy)
		new_policy = policy_improvement(V, old_policy)
		if policy == new_policy:
			print('Policy-Iteration converged at step %d.' %(i+1))
			break
		policy = new_policy
	return policy, V

def policy_evaluation(policy):
	# V = np.zeros(env.nS)
	V = {}
	for s in env.P:
		V[s] = 0.0

	while True:
		delta = 0
		for s in env.P:
			total_state_value = 0.0
			for a, prob_a in policy[s].items():
				prob_s, next_state, reward, is_done = env.get_next_state_info_upon_action(a, s)
				if is_done:
					total_state_value += prob_a * prob_s * (reward + gamma * V[next_state])
				else:
					_, opp_next_state, _, opp_is_done = env.get_next_state_info_following_policy(policy, next_state) # opponent acts and return a new state
					if not opp_is_done:
						# pdb.set_trace()
						total_state_value += prob_a * prob_s * (reward + gamma * V[opp_next_state])

			delta = max(delta, np.abs(total_state_value - V[s]))
			V[s] = total_state_value
		# print(f'delta: {delta}')
		if delta < 0.005:
			break	
	return V

def policy_improvement(V, policy):
	for s, v in env.P.items():
		Q_sa = {}
		for a in v:
			Q_sa[a] = 0.0
		# print(Q_sa)
		for a in v:
			# print(f'{s} {a}')
			next_state = None
			prob_s, next_s, reward, is_done = env.get_next_state_info_upon_action(a, s)
			if is_done:
				next_state = next_s
			else:
				_, opp_next_state, _, opp_is_done = env.get_next_state_info_following_policy(policy, next_s)
				# if not opp_is_done:
				next_state = opp_next_state

			Q_sa[a] = prob_s * (reward + gamma * V[next_state])
		# print(Q_sa)
		if Q_sa:
			best_action = max(Q_sa, key=Q_sa.get)
		else:
			best_action = None
		for a in v:	
			if best_action == a:
				policy[s][a] = 1.0
			else:
				policy[s][a] = 0.0

	return policy

p, v = policy_iteration()
optimal_policy = env.get_best_actions(p)
print(optimal_policy)
with open('policy.txt', 'w') as f:
	f.write('\n'.join([f'{k[0]} {k[1]} {k[2]} {k[3]}\t{v}' for k, v in optimal_policy.items()]))


