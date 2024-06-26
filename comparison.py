from __future__ import annotations


VOWEL = {'a', 'i', 'e', 'o', 'u'}

def division(s:str):
    for i in range(len(s)):
        if s[i] in VOWEL:
            return s[:i], s[i:]


class Syllable:
    def __init__(self, rom:str, jap:str) -> None:
        self.consonant = ''
        self.vowel = ''
    
    def comparison(self, other:Syllable) -> float:
        pass
    
    def print():
        pass

