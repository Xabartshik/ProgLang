from lib.grammarSolver import GrammarSolver



# Данные для задачи 2а
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

# Создание экземпляров GrammarSolver и определение типов
solver_2a = GrammarSolver(rules_2a, nonterminals_2a, terminals_2a, start_symbol_2a)
solver_2b = GrammarSolver(rules_2b, nonterminals_2b, terminals_2b, start_symbol_2b)

# Вывод типов грамматик
print("Тип грамматики для задачи 2а:", solver_2a.get_grammar_type())
print("Тип грамматики для задачи 2б:", solver_2b.get_grammar_type())
