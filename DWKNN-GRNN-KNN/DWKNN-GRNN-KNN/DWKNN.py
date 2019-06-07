import math
import numpy as np
import random

runner = -1

def generate_do(author, limit, dos):
    do = [0] * limit
    do[author] = dos[author]
    return do


def get_author_from_list(author_list):
    return np.argmax(author_list)


def dist_sqrd(t_q, t_i):
    dist_sum = 0.0
    for i in range(0, len(t_q)):
        dist_sum += math.pow((t_q[i] - t_i[i]), 2.0)
    return math.sqrt(dist_sum)


def get_weight(dist, b):
    return 1/math.pow(dist, b)


def get_neighbors(query, tset, k):
    closest = []
    for index, value in enumerate(tset):
        if dist_sqrd(value[0], query[0]) != 0:
            closest.append([
                dist_sqrd(value[0], query[0]),
                value[1]
            ])

    closest = sorted(closest, key=lambda x: x[0], reverse=False)[:k]

    for i, item in enumerate(closest):
        closest[i][0] = get_weight(closest[i][0], 5)
    return closest


# train  [ Instance => [0] => feature vector, [1] => desired output ]
def leave_one_out(train, k):
    error = 0
    for index, value in enumerate(train):
        test = value
        # get all instances except the selected one
        t = [x for i, x in enumerate(train) if i != index and (runner != 2 or x[2] != test[2])]
        if k == 7:
            k = len(t)
        neighbors = get_neighbors(test, t, k)
        sum_weight = sum(i[0] for i in neighbors)    # sum of all weights
        sum_values = sum(i[0] * np.array(i[1]) / sum_weight for i in neighbors) # sum of d*weight
        # print(value[1], "=>", neighbors, sumValues, np.argmax(sumValues))
        winning_author = np.argmax(sum_values)
        if winning_author != get_author_from_list(value[1]):
            # print("Winner:"+str(winningAuthor)+", for:", index, "expected:", getAuthorFromList(value[1]), sumValues)
            # print("for:" ,index, ":", str(winningAuthor) ,"-" ,getAuthorFromList(value[1]), neighbors)
            error += 1
    error_rate = error/len(train)
    print("K = " + str(k) + " => error:", error_rate*100, " - success", (1-error_rate)*100)
    return (1-error_rate)*100


def read_instances(runner):
    if runner == 0:
        file_name = 'results_tsncu.txt'
    else:
        file_name = 'msst_tsncu.txt'

    file_stream = open(file_name, 'r')
    train = []
    authors = []

    for line in file_stream:
        cols = line.strip().split(',')
        author = cols.pop(0)
        match = -1
        for index, value in enumerate(cols):
            cols[index] = float(value)
        if runner == 0:
            author = int(author.split('_')[0]) - 1000
        else:
            #print(author)
            sepa = author.split('_')
            author = sepa[0] + sepa[1]
            match = sepa[2] + sepa[3]

        if author not in authors:
            authors.append(author)
        train.append([cols, author, match])

    return train, authors

first = 1


for run in range(0, 3):
    runner = run
    traino, authorso = read_instances(runner)
    bestDos = [
        [0, []],
        [0, []],
        [0, []],
        [0, []]
    ]
    for j in range(0, 1):
        #if j % (1000/10) == 0:
        #print("Iteration: ", j)
        train, authors = read_instances(runner)
        if runner == 2:
            dos = [1.9702375022466017, -0.029941412367614717, -0.027458428165161752, 1.6167351424618774, 0.7864578406309055, -0.6760535926260818, -0.06525358601277209] # [random.gauss(1, 0.85) for i in range(0, len(authors))]
        else:
            dos = [1] * len(authors)

        for index, value in enumerate(train):
            author_index = authors.index(value[1])
            train[index][1] = generate_do(author_index, len(authors), dos)

        print("Running for ", runner)
        results = []
        for k in range(1, 8, 2):
            results.append(leave_one_out(train, k))

        result = np.mean(results)
        if bestDos[runner][0] < result:
            bestDos[runner][1] = dos
            bestDos[runner][0] = result

print("Best doses:", bestDos)


