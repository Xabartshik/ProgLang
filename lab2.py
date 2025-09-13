from lib.grammarSolver import GrammarSolver

nonterminals_2a = {'S', 'A', 'B', 'C', 'D', 'F'}
terminals_2a = {'a', 'b', 'c'}
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
start_symbol_2a = 'S'

# Данные для задачи 2б
nonterminals_2b = {'S', 'A', 'B'}
terminals_2b = {'a', 'b', '1'}
rules_2b = {
    'S': ['A1', 'B1'],
    'A': ['a', 'Ba'],
    'B': ['b', 'Bb', 'Ab']
}
start_symbol_2b = 'S'

# Создаём и выводим результаты
solver_2a = GrammarSolver(rules_2a, nonterminals_2a, terminals_2a, start_symbol_2a)
solver_2b = GrammarSolver(rules_2b, nonterminals_2b, terminals_2b, start_symbol_2b)

print("=== Задача 2а ===")
solver_2a.print_grammar()
print("Тип грамматики:", solver_2a.get_grammar_type_name())
desc2a = solver_2a.generate_description()
lang2a = solver_2a.generate_language(method='exhaustive', max_length=10, max_strings=10)
print(solver_2a.format_language_output("G2a", desc2a, lang2a))

print("=== Задача 2б ===")
solver_2b.print_grammar()
print("Тип грамматики:", solver_2b.get_grammar_type_name())
desc2b = solver_2b.generate_description()
lang2b = solver_2b.generate_language(method='exhaustive', max_length=10, max_strings=10)
print(solver_2b.format_language_output("G2b", desc2b, lang2b))
