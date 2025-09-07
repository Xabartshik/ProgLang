from lib.grammarSolver import GrammarSolver



# Множество нетерминальных символов
nonterminals_2a = {'S', 'A', 'B', 'C', 'D', 'F'}

# Множество терминальных символов
terminals_2a = {'a', 'b', 'c'}

# Правила грамматики
rules_2a = {
    'S': ['aaCFD'],
    'AD': ['D'],
    'F': ['AFB', 'AB'],
    'Cb': ['bC'],
    'AB': ['bBA'],
    'CB': ['C'],
    'Ab': ['bA'],
    'bCD': ['ε']
}

# Начальный символ
start_symbol_2a = 'S'

# Множество нетерминальных символов
nonterminals_2b = {'S', 'A', 'B'}

# Множество терминальных символов
terminals_2b = {'a', 'b', '1'}

# Правила грамматики
rules_2b = {
    'S': ['A1', 'B1'],
    'A': ['a', 'Ba'],
    'B': ['b', 'Bb', 'Ab']
}

# Начальный символ
start_symbol_2b = 'S'

# Тест для 2а
print("Язык 2а (max_depth=3):")
language_2a, is_infinite_2a = GrammarSolver.generate_language(rules_2a, nonterminals_2a, terminals_2a, start_symbol_2a, max_depth=10)
for string in sorted(language_2a):
    print(string)
print(f"Бесконечный: {is_infinite_2a}")

# Тест для 2б
print("\nЯзык 2б (max_depth=3):")
language_2b, is_infinite_2b = GrammarSolver.generate_language(rules_2b, nonterminals_2b, terminals_2b, start_symbol_2b, max_depth=10)
for string in sorted(language_2b):
    print(string)
print(f"Бесконечный: {is_infinite_2b}")
