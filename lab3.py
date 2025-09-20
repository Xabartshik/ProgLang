import sys
import datetime

from io import StringIO
from lib.grammarSolver import GrammarSolver

class FileLogger:
    """Класс для одновременного вывода в консоль и файл"""
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout

        # Записываем заголовок файла
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.file.write("=" * 80 + "\n")
        self.file.write("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ LAB3\n")
        self.file.write("=" * 80 + "\n")
        self.file.write(f"Дата выполнения: {timestamp}\n\n")

    def write(self, text):
        # Выводим в консоль
        self.original_stdout.write(text)
        # Записываем в файл
        self.file.write(text)

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.original_stdout


# Запускаем логирование
logger = FileLogger("lab3_results.txt")
sys.stdout = logger

try:
    print("ЛАБОРАТОРНАЯ РАБОТА 3")
    print("Анализ и преобразование формальных грамматик")
    print("=" * 60)

    # === ЗАДАЧА 6 ===
    print("\n=== ЗАДАЧА 6 ===")
    print("Упрощение грамматики и сравнение языков")
    print("-" * 40)

    # Исходные правила для задачи 6
    rules_6 = {
        'S': ['AB', 'ABS'],
        'AB': ['BA'],
        'BA': ['AB'],
        'A': ['a'],
        'B': ['b']
    }

    # Создаем решатели
    solver_6 = GrammarSolver(rules_6)
    solver_6_control = GrammarSolver(rules_6)

    print("Исходная грамматика:")
    solver_6_control.print_grammar()

    # Упрощаем грамматику
    solver_6.simplify_grammar()
    print("\nУпрощенная грамматика:")
    solver_6.print_grammar()

    # Сравнение языков
    print("\nСравнение языков до и после упрощения (максимальная длина 10):")
    GrammarSolver.compare_languages(solver_6, solver_6_control, n=10, method="recursive")

    # === ЗАДАЧА 8 ===
    print("\n=== ЗАДАЧА 8 ===")
    print("Генерация языка do-while-операторов")
    print("-" * 40)

    # # Правила и алфавиты для do-while
    # rules_8 = {
    #     'do-while-statement': ['do statement while ( expression ) ;'],
    #     'statement': ['{ statements }', 'single_statement'],
    #     'statements': ['statement', 'statement statements'],
    #     'single_statement': ['expression ;'],
    #     'expression': ['identifier', 'constant']
    # }
    # Правила и алфавиты для do-while
    # nonterminals_8 = {'do-while-statement', 'statement', 'statements', 'single_statement', 'expression'}
    # terminals_8 = {'do', 'while', '(', ')', ';', '{', '}', 'identifier', 'constant'}
    # start_symbol_8 = 'do-while-statement'
    #
    # solver_8 = GrammarSolver(
    #     rules_8,
    #     nonterminals=nonterminals_8,
    #     terminals=terminals_8,
    #     start_symbol=start_symbol_8
    # )
    #
    # # Генерация строк языка
    # lang8 = solver_8.generate_language_with_spaces_support(
    #     max_length=100,
    #     max_strings=200,
    #     max_depth=100,
    #     require_all_terminals=True
    # )
    #
    # print("\nПримеры строк языка:")
    # for i, s in enumerate(lang8, 1):
    #     display = s if s else 'ε'
    #     print(f" {i:2d}. '{display}' (длина: {len(s)})")
    #
    # print("\n=== ЗАВЕРШЕНИЕ LAB3 ===")
    # print("=" * 60)
    rules_8 = {
        'D': ['dSw(E);'],  # D - do-while-statement, d - do, w - while
        'S': ['{T}', 's'],  # S - statement, T - statements, s - single_statement
        'T': ['S', 'ST'],
        's': ['E;'],
        'E': ['i', 'c']  # E - expression, i - identifier, c - constant
    }
    nonterminals_8 = {'D', 'S', 'T', 's', 'E'}
    terminals_8 = {'d', 'w', '(', ')', ';', '{', '}', 'i', 'c'}
    start_symbol_8 = 'D'

    solver_8 = GrammarSolver(
        rules_8,
        nonterminals=nonterminals_8,
        terminals=terminals_8,
        start_symbol=start_symbol_8
    )

    # Генерация строк языка
    lang8 = solver_8.generate_language(
        max_length=100,
        max_strings=10,
        max_depth=10,
        require_all_terminals=True
    )

    # Обратная замена символов на исходные выражения
    symbol_to_expression = {
        'D': ' do-while-statement ',
        'S': ' statement ',
        'T': ' statements ',
        's': ' single_statement ',
        'E': ' expression ',
        'd': ' do ',
        'w': ' while ',
        'i': ' identifier ',
        'c': ' constant ',
        '(': '(',
        ')': ')',
        ';': ';',
        '{': '{',
        '}': '}'
    }

    # Замена символов в сгенерированных строках
    restored_lang8 = []
    for s in lang8:
        restored = ''
        for char in s:
            restored += symbol_to_expression[char]
        restored_lang8.append(restored.strip())

    print("\nПримеры строк языка:")
    for i, s in enumerate(restored_lang8, 1):
        display = s if s else 'ε'
        print(f" {i:2d}. '{display}' (длина: {len(s)})")

    print("\n=== ЗАВЕРШЕНИЕ LAB3 ===")
    print("=" * 60)

finally:
    # Закрываем логгер и восстанавливаем stdout
    logger.close()
