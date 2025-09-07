# Пример использования для задания 1a
from lib.grammarSolver import GrammarSolver

rules_1a = {
    'S': ['T+S', 'T-S', 'T'],
    'T': ['F*T', 'F'],
    'F': ['a', 'b']
}
nonterm_1a = {'S', 'T', 'F'}
term_1a = {'a', 'b', '+', '-', '*'}
solver_1a = GrammarSolver(rules_1a, nonterm_1a, term_1a)
print("Вывод для 1a 'a-b*a+b':")
print('\n'.join(solver_1a.derive_string('a-b*a+b')))

# Пример использования для задания 1b
rules_1b = {
    'S': ['aSBC', 'abC'],
    'CB': ['BC'],
    'bB': ['bb'],
    'bC': ['bc'],
    'cC': ['cc']
}
nonterm_1b = {'S', 'B', 'C'}
term_1b = {'a', 'b', 'c'}
solver_1b = GrammarSolver(rules_1b, nonterm_1b, term_1b)
print("\nВывод для 1b 'aaabbbccc':")
print('\n'.join(solver_1b.derive_string('aaabbbccc')))
rules = {'S': ['aaCFD', 'b'], 'A': ['a']}
nonterm = {'S', 'A'}
term = {'a', 'b'}
grammar = GrammarSolver(rules, nonterm, term)
print(GrammarSolver.generate_language(grammar, 'S'))