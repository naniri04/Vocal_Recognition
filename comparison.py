'''
similarity_matrix -> find_opt
'''

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    csn_eq = (org_csn_mat == gen_csn_mat).astype(int) * 0.33
    csn_sim = (csn_get(org_csn_mat) == csn_get(gen_csn_mat)).astype(int) * 0.25
    vwl_eq = (org_vwl_mat == gen_vwl_mat).astype(int) * 0.33
    vwl_sim = (np.vectorize(first_eq)(org_vwl_mat, gen_vwl_mat)).astype(int) * 0.25
    score_mat = np.maximum(csn_eq, csn_sim) + np.maximum(vwl_eq, vwl_sim)
    score_mat[score_mat == 0.66] = 1
    
    return score_mat


def mat_heatmap(mat, points=0, figsize=(8,6), ps=1):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(mat, cmap='hot', interpolation='nearest', aspect='equal'
                    )
    fig.colorbar(cax)
    ax.set_xticks(np.arange(0, mat.shape[1], 4))
    ax.set_yticks(np.arange(0, mat.shape[0], 4))
    c, r = zip(*points) if points else ([], [])
    if points: ax.scatter(r, c, s=ps, alpha=0.5, marker='s')
    plt.show()
    
import time
    
def find_opt(sim_mat) -> tuple[float, list, np.ndarray]:
    n, m = sim_mat.shape
    print(n, 'x', m)
    PENALTY = 0.6

    def greedy(dr, dc):  # dr, dc는 index
        grid = np.zeros((n, dc+1)); pos_to = np.zeros((n, dc+1), dtype=np.int8)
        cut_sim = sim_mat[:dr+1, :dc+1]
        grid[-1, -1] = cut_sim[dr, dc]
        pos_to[-1, -1] = 3
        
        def get_val(r, c):
            val_list = [grid[r, c+1] + cut_sim[r, c] - PENALTY
                    , grid[r+1, c] + cut_sim[r, c] - PENALTY
                    , grid[r+1, c+1] + cut_sim[r, c]]
            if val_list[0] > val_list[1]:
                pos_to[r, c] = 0 if val_list[0] > val_list[2] else 2
            else:
                pos_to[r, c] = 1 if val_list[1] > val_list[2] else 2
            grid[r, c] = val_list[pos_to[r, c]]            
        
        for i in range(2, max(n, dc+1)+1):
            if i <= dc+1:  # 세로
                grid[-1, -i] = grid[-1, -i+1] + cut_sim[-1, -i] - PENALTY
                pos_to[-1, -i] = 0
                for mr in range(2, min(i, dr+2)): get_val(-mr, -i)
            if i <= n:  # 가로
                grid[-i, -1] = grid[-i+1, -1] + cut_sim[-i, -1] - PENALTY
                pos_to[-i, -1] = 1
                for mc in range(2, min(i, dc+2)): get_val(-i, -mc)
            if i <= n and i <= dc+1:
                get_val(-i, -i)
        
        max_pos = np.argmax(np.concatenate([grid[0], grid[:,0]]))
        max_pos = [0, max_pos] if max_pos < dc else [max_pos-dc-1, 0]
        path = [(max_pos[0], max_pos[1])]
        while True:
            if pos_to[max_pos[0], max_pos[1]] == 3: break
            future_pos = pos_to[max_pos[0], max_pos[1]]
            if future_pos in (0,2): max_pos[1] += 1
            if future_pos in (1,2): max_pos[0] += 1
            path.append((max_pos[0], max_pos[1]))
        #
        return grid[path[0][0], path[0][1]], path, grid
    
    max_val = -1; path = []; grid = []
    for j in tqdm(range(int(m*0.92), m)):
        cval, cpath, cgrid = greedy(n-1, j)
        # mat_heatmap(grid, cpath)
        if cval > max_val:
            max_val = cval; path = cpath; grid = cgrid
    # max_val, path, grid = greedy(n-1, m-1)
        
    return max_val, path, grid
