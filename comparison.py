'''
similarity_matrix -> find_opt
'''

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import heapq

VOWEL = {'a', 'i', 'e', 'o', 'u', 'y'}
EXC_CSN = {'ch', 'sh', 'ts'}
SIMILAR_CSN = [
    {'b', 'f', 'p', 'v'},
    {'c', 'g', 'k', 'q'},
    {'d', 't'},
    {'l'},
    {'m', 'n'},
    {'r'},
    {'s', 'sh', 'ts', 'z', 'x'},
    {'w'},
    {'h'},
    {'ch', 'j'},
    {''}
]
csn_sep = {}
for idx, ks in enumerate(SIMILAR_CSN):
    for k in ks: csn_sep[k] = idx
for si in [str(i) for i in range(10)]:
    csn_sep[si] = si


def division(s:str):
    if s.isnumeric(): return s, s
    for i in range(len(s)):
        if s[i] in VOWEL:
            return s[:i], s[i:]
    return s, '0'


class Syllable:
    def __init__(self, rom:str, jap:str=None) -> None:
        self.consonant, self.vowel = division(rom)
        self.rom = rom
        self.jap = jap
    
    def sim(self, other:Syllable) -> float:
        score = 0
        # consonant
        if self.consonant == other.consonant: score += 0.5
        elif csn_sep[self.consonant] == csn_sep[other.consonant]: score += 0.25
        # vowel
        if self.vowel == other.vowel: score += 0.5
        elif self.vowel[0] == other.vowel[0]: score += 0.25
        #
        return score
        
    def __repr__(self) -> str:
        return self.rom


def seperation(sentence:str) -> list[Syllable]:
    sentence = sentence.replace(' ', '')
    sl = []; prev = [0]; flag = 0
    def ap(i):
        sl.append(Syllable(sentence[prev[0]:i]))
        prev[0] = i
    def ex_check(ch, idx):
        return (ch == 't' and sentence[idx] == 's') or \
        (ch == 'c' and sentence[idx] == 'h') or (ch == 's' and sentence[idx] == 'h')
    
    for idx, ch in enumerate(sentence, start=1):
        # print(flag, ch)
        if flag == 1:
            flag = 0; ap(idx); continue
        elif flag == 2:
            flag = 0; continue
        #
        if ch.isnumeric(): ap(idx)
        elif ch == "ー":
            prev[0] -= 1
            ap(idx-1)
            prev[0] = idx
        elif ch in VOWEL:
            if ch == 'y': flag = 1
            else: ap(idx)
        else:
            if flag == 3:
                flag = 0; ap(idx-1)
            #
            if ex_check(ch, idx): flag = 2
            else: flag = 3; continue
        
        if flag == 3: flag = 0
    
    if flag == 3:
        ap(len(sentence))
    
    return sl


def similarity_matrix(original:str, generated:str) -> np.ndarray:
    def first_eq(x, y): return x[0] == y[0]

    org, gen = seperation(original), seperation(generated)
    csn_get = np.vectorize(csn_sep.get)
    matgen = lambda x, y, b: \
        np.array([np.tile(sy.consonant if b else sy.vowel, len(y)) for sy in x])
    org_csn_mat, org_vwl_mat = matgen(org, gen, True), matgen(org, gen, False)
    gen_csn_mat, gen_vwl_mat = matgen(gen, org, True).T, matgen(gen, org, False).T

    csn_eq = (org_csn_mat == gen_csn_mat).astype(int) * 0.5
    csn_sim = (csn_get(org_csn_mat) == csn_get(gen_csn_mat)).astype(int) * 0.25
    vwl_eq = (org_vwl_mat == gen_vwl_mat).astype(int) * 0.5
    vwl_sim = (np.vectorize(first_eq)(org_vwl_mat, gen_vwl_mat)).astype(int) * 0.25
    score_mat = np.maximum(csn_eq, csn_sim) + np.maximum(vwl_eq, vwl_sim)
    
    return score_mat


def mat_heatmap(mat, points=0):
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap')
    plt.xticks(np.arange(0, mat.shape[1], 4))
    plt.yticks(np.arange(0, mat.shape[0], 4))
    c, r = zip(*points) if points else ([], [])
    if points: plt.scatter(r, c, s=1)
    plt.show()
    
    
def find_opt(sim_mat) -> tuple[float, list]:
    n, m = sim_mat.shape
    print(n, 'x', m)

    dp = np.ones((n, m)); visited = np.zeros((n, m))
    path = [[[]]*m for _ in range(n)]; q = []
    for i in range(n):
        q.append((-sim_mat[i,0], i, 0, []))
        dp[i,0] = -sim_mat[i,0]
    for i in range(1, m):
        q.append((-sim_mat[0,i], 0, i, []))
        dp[0,i] = -sim_mat[0,i]
    heapq.heapify(q)
    
    PENALTY = 1
    while q:
        acc, r, c, p = heapq.heappop(q)
        if not visited[r, c]:
            visited[r, c] = 1
            path[r][c] = p+[(r, c)]
        else: continue
        
        for tr, tc, pen in [(r+1,c+1,0), (r,c+1,PENALTY), (r+1,c,PENALTY)]:
            if tr >= n or tc >= m: continue
            val = acc - sim_mat[tr, tc] + pen
            if dp[tr, tc] > val:
                heapq.heappush(q, (val, tr, tc, path[r][c]))
                dp[tr, tc] = val
            
    dp *= -1
    print(dp)
    mc = np.argmax(dp[-1])
    max_sum = dp[-1][mc]
    max_path = path[-1][mc]

    print("Maximum path sum:", max_sum)
    print("Accuracy: ", max_sum / min(*sim_mat.shape))
    print("Path:", max_path)

    return max_sum, max_path
