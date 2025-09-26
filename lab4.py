import sys
import datetime

from lib.CFGParser import CFGParser


class FileLogger:
    """Класс для одновременного вывода в консоль и файл"""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout

        # Записываем заголовок файла
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.file.write("=" * 80 + "\n")
        self.file.write("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА\n")
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
logger = FileLogger("lab4_results.txt")
sys.stdout = logger

try:
    # Пример использования для данной грамматики
    rules = {
        'S': ['aSbS', 'bSaS', '']
    }
    terminals = {'a', 'b'}
    nonterminals = {'S'}

    parser = CFGParser(rules, nonterminals, terminals)

    print("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА")
    print("=" * 60)
    target_string = "abab"
    parser.print_parse_trees(target_string)

    # Пример использования для задачи 11а
    print("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА И ПРЕОБРАЗОВАНИЕ ГРАММАТИК")
    print("=" * 60)

    # Грамматика 11а
    rules_a = {
        'S': ['0S', '0B'],
        'B': ['1B', '1C'],
        'C': ['1C', '$']
    }
    terminals_a = {'0', '1', '$'}
    nonterminals_a = {'S', 'B', 'C'}

    parser_a = CFGParser(rules_a, nonterminals_a, terminals_a)

    # Увеличиваем лимит рекурсии (временная мера, если нужно)
    sys.setrecursionlimit(2000)

    print("\n=== Задача 11а ===")
    print("Исходная праволинейная грамматика:")
    parser_a.print_grammar()

    left_linear_a = parser_a.to_left_linear_grammar()
    parser_a.print_grammar()

    # Тестируем деревья вывода для цепочки
    target_string = "001$"
    parser_a.print_parse_trees(target_string)

    # Грамматика 11б
    rules_b = {
        'S': ['aA', 'aB', 'bA'],
        'A': ['bS'],
        'B': ['aS', 'bB', '$']
    }
    terminals_b = {'a', 'b', '$'}
    nonterminals_b = {'S', 'A', 'B'}

    parser_b = CFGParser(rules_b, nonterminals_b, terminals_b)

    print("\n=== Задача 11б ===")
    print("Исходная праволинейная грамматика:")
    parser_b.print_grammar()

    parser_b.rules = parser_b.to_left_linear_grammar()
    parser_b.print_grammar()

    # Тестируем деревья вывода для цепочки
    target_string_b = "ab$"
    parser_b.print_parse_trees(target_string_b)


    print("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА И ПРЕОБРАЗОВАНИЕ ГРАММАТИК")
    print("=" * 60)

    rules_g1 = {
        'S': ['S1', 'A0'],
        'A': ['A1', '0']
    }
    terminals_g1 = {'0', '1'}
    nonterminals_g1 = {'S', 'A'}
    parser_g1 = CFGParser(rules_g1, nonterminals_g1, terminals_g1)

    rules_g2 = {
        'S': ['A1', 'B0', 'E1'],
        'A': ['S1'],
        'B': ['C1', 'D1'],
        'C': ['0'],
        'D': ['B1'],
        'E': ['E0', '1']
    }
    terminals_g2 = {'0', '1'}
    nonterminals_g2 = {'S', 'A', 'B', 'C', 'D', 'E'}
    parser_g2 = CFGParser(rules_g2, nonterminals_g2, terminals_g2)

    print("\n=== Задача 12 ===")
    print("Грамматика G1:")
    parser_g1.print_grammar_nice(rules_g1, "Леволинейная грамматика G1")

    print("\nГрамматика G2:")
    parser_g2.print_grammar_nice(rules_g2, "Леволинейная грамматика G2")

    parser_g1.rules = parser_g1.to_right_linear_grammar()
    parser_g1.print_grammar_nice(parser_g1.rules, "Праволинейная грамматика G1")

    parser_g2.rules = parser_g2.to_right_linear_grammar()
    parser_g2.print_grammar_nice(parser_g2.rules, "Праволинейная грамматика G2")


    print("\n=== ЗАВЕРШЕНИЕ ===")
    print("=" * 60)

finally:
    # Закрываем логгер и восстанавливаем stdout
    logger.close()