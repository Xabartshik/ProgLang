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
        self.file.write("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ LAB2\n")
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
logger = FileLogger("lab2_results.txt")
sys.stdout = logger

try:
    print("ЛАБОРАТОРНАЯ РАБОТА 2")
    print("Анализ формальных грамматик и генерация языков")
    print("=" * 60)

    # Данные для задачи 2а
    nonterminals_2a = {'S', 'A', 'B', 'C', 'D', 'F', 'AD', 'AB', 'Cb', 'CB', 'Ab', 'bCD'}
    terminals_2a = {'a', 'b'}
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

    print("\n=== ЗАДАЧА 2а ===")
    print("Анализ контекстно-зависимой грамматики")
    print("-" * 40)
    solver_2a.print_grammar()
    print("Тип грамматики:", solver_2a.get_grammar_type_name())
    desc2a = solver_2a.generate_description()
    print(f"Описание языка: L(G2a) = {{{desc2a}}}")

    lang2a = solver_2a.generate_language_with_all_terminals(method='exhaustive', max_length=20, max_strings=50)
    print(f"\nНайдено строк языка: {len(lang2a)}")
    print("Примеры строк:")
    for i, s in enumerate(lang2a, 1):
        display = s if s else 'ε'
        print(f"  {i:2d}. '{display}' (длина: {len(s)})")

    print("\n=== ЗАДАЧА 2б ===")
    print("Анализ регулярной грамматики")
    print("-" * 40)
    solver_2b.print_grammar()
    print("Тип грамматики:", solver_2b.get_grammar_type_name())
    desc2b = solver_2b.generate_description()
    print(f"Описание языка: L(G2b) = {{{desc2b}}}")

    lang2b = solver_2b.generate_language_with_all_terminals(method='exhaustive', max_length=10, max_strings=20)
    print(f"\nНайдено строк языка: {len(lang2b)}")
    print("Примеры строк:")
    for i, s in enumerate(lang2b, 1):
        display = s if s else 'ε'
        print(f"  {i:2d}. '{display}' (длина: {len(s)})")

    print("\n=== ЗАДАЧА 3б ===")
    print("Создание грамматики по описанию языка L = {0^n(10)^m | n, m ≥ 0}")
    print("-" * 40)

    # Терминалы языка
    terminals_3b = {'0', '1'}

    # Формальное описание языка
    description_3b = "0^n(10)^m | n, m ≥ 0"

    # Генерация грамматики по описанию
    grammar_3b = GrammarSolver.from_language_description(terminals_3b, description_3b)

    # Вывод грамматики
    print("Автоматически созданная грамматика:")
    grammar_3b.print_grammar()
    print(f"Описание языка: L(G3b) = {{{grammar_3b.generate_description()}}}")
    print(f"Тип грамматики: {grammar_3b.get_grammar_type_name()}")

    # Примеры строк языка
    strings_3b = grammar_3b.generate_language_with_all_terminals(method='exhaustive', max_length=12, max_strings=15)
    print(f"\nНайдено строк языка: {len(strings_3b)}")
    print("Примеры строк языка:")
    for i, s in enumerate(strings_3b, 1):
        display = s if s else 'ε'
        print(f"  {i:2d}. '{display}' (длина: {len(s)})")

    print("\n=== ЭКСПОРТ ПОДРОБНЫХ АНАЛИЗОВ ===")
    print("Создание файлов детального анализа...")

    # Экспортируем каждую грамматику в отдельный файл
    solver_2a.export_to_file("analysis_task_2a.txt")
    print("✓ Создан файл: analysis_task_2a.txt")

    solver_2b.export_to_file("analysis_task_2b.txt")
    print("✓ Создан файл: analysis_task_2b.txt")

    grammar_3b.export_to_file("analysis_task_3b.txt")
    print("✓ Создан файл: analysis_task_3b.txt")

    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЛАБОРАТОРНОЙ РАБОТЫ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Все результаты сохранены в файлы:")
    print("- lab2_results.txt (основные результаты)")
    print("- analysis_task_2a.txt (детальный анализ задачи 2а)")
    print("- analysis_task_2b.txt (детальный анализ задачи 2б)")
    print("- analysis_task_3b.txt (детальный анализ задачи 3б)")

finally:
    # Закрываем файл и восстанавливаем stdout
    logger.close()
