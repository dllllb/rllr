import fileinput

with open('/home/vladimir/temp.txt', 'r') as r:
    Lines = r.readlines()

i = 0
T = None
P, Q, j, task = None, None, 0, []
tasks = []
for line in Lines: #fileinput.input():
    if i == 0:
        T = int(line)
        i += 1
        continue
    if P is None:
        P, Q = line.split()
        P, Q = int(P), int(Q)
        task.append(P)
        task.append(Q+1) # since coords are from [0, Q]
        j = 0
        continue
    if j < P:
        x, y, d = line.split()
        x, y = int(x), int(y)
        task.append((x, y, d))
        j += 1
        if j == P:
            P, Q, j = None, None, 0
            tasks.append(task)
            task = []

def solve_task(task_id, task):
    P, Q = task[0], task[1]
    n, s, e, w = [0]*Q, [0]*Q, [0]*Q, [0]*Q
    for i in range(P):
        x, y, d = task[2 + i]
        if d == 'N':
            n[y+1] = 1
        if d == 'S':
            s[y+1] = 1
        if d == 'E':
            e[x+1] = 1
        if d == 'W':
            w[x+1] = 1

    for j in range(1, Q):
        n[j] += n[j-1]
        s[j] += s[j-1]
        w[j] += w[j-1]
        e[j] += e[j-1] 

    for j in range(Q):
        n[j] += s[Q-1-j]
        w[j] += e[Q-1-j]

    max_w, max_n = w[Q-1], n[0]
    x, y = Q-1, 0
    for j in range(Q-1, -1, -1):
        if w[j] >= max_w:
            max_w = w[j]
            x = j
        if n[Q-1-j] >= max_n:
            max_n = n[Q-1-j]
            y = Q-1-j

    return f'Case #{task_id}: {x} {y}'

for i, task in enumerate(tasks):
    out = solve_task(i+1, task)
    print(out)