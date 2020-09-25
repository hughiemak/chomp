with open('policy.txt', 'r') as f:
	content = [l.rstrip().split('\t') for l in f]
content = [[l[0].split(' '), l[1]]for l in content]
content = [[[int(x) for x in l[0]], int(l[1])] for l in content]
H = 4
W = 7

S = []

for h in range(W+1):
	for i in range(W+1):
		for j in range(W+1):
			for k in range(W+1):
				if k >= j and j >= i and i >= h:
					# print((k,j,i,h))
					S.append([k,j,i,h])

out = []
for s in S:
	x = [x for x in content if x[0] == s]
	if x:
		out.append(x[0])

with open('policy_ordered.txt', 'w') as f:
	f.write('\n'.join([f'{x[0][0]} {x[0][1]} {x[0][2]} {x[0][3]}\t{x[1]}' for x in out]))