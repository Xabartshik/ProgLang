import sys
import datetime
from typing import List, Dict, Set, Optional


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


class CFGParser:
    def __init__(self, rules: Dict[str, List[str]], nonterminals: Optional[Set[str]] = None,
                 terminals: Optional[Set[str]] = None, start_symbol: str = 'S'):
        """
        :param rules: Словарь продукций: ключ - левая часть (str), значение - список правых частей (str).
        :param nonterminals: Множество нетерминальных символов (необязательно - будет определено автоматически).
        :param terminals: Множество терминальных символов (необязательно - будет определено автоматически).
        :param start_symbol: Стартовый символ.
        """
        self.rules = rules
        self.start_symbol = start_symbol

        if nonterminals is None:
            self.nonterminals = set(rules.keys())
        else:
            self.nonterminals = nonterminals

        if terminals is None:
            self.terminals = set()
            for productions in rules.values():
                for prod in productions:
                    for symbol in prod:
                        if symbol not in self.nonterminals:
                            self.terminals.add(symbol)
        else:
            self.terminals = terminals

    def count_terminals(self, string: str) -> int:
        """Подсчитывает количество терминальных символов в строке."""
        return sum(1 for c in string if c in self.terminals)

    def is_possible_prefix(self, current: str, target: str) -> bool:
        """
        Проверяет, может ли текущая строка (без нетерминалов) быть префиксом целевой.
        """
        current_clean = ''.join(c for c in current if c in self.terminals)
        if len(current_clean) > len(target):
            return False
        return current_clean == target[:len(current_clean)]

    def generate_parse_trees(self, target: str, current_string: Optional[str] = None,
                             derivation: Optional[List[str]] = None) -> List[List[str]]:
        """
        Генерирует все возможные деревья вывода для целевой цепочки.
        :param target: Целевая строка.
        :param current_string: Текущая строка на этапе вывода (по умолчанию start_symbol).
        :param derivation: Список шагов вывода (для построения дерева).
        """
        if current_string is None:
            current_string = self.start_symbol

        if derivation is None:
            derivation = [f"Start: {self.start_symbol}"]

        # Базовый случай: если текущая строка совпадает с целевой и не осталось нетерминалов
        if current_string == target and all(c in self.terminals for c in current_string):
            return [derivation]

        # Проверяем, возможна ли целевая строка
        if not self.is_possible_prefix(current_string, target):
            return []

        # Проверяем, не слишком ли много терминалов
        if self.count_terminals(current_string) > len(target):
            return []

        results = []

        # Перебираем все позиции в текущей строке
        i = 0
        while i < len(current_string):
            symbol = current_string[i]
            if symbol in self.nonterminals:
                # Применяем каждое правило для этого нетерминала
                for prod in self.rules.get(symbol, []):
                    # Заменяем символ на продукцию (пустая строка для ε)
                    new_string = current_string[:i] + prod + current_string[i + 1:]
                    step = f"{symbol} -> {prod if prod else 'ε'} at pos {i}"
                    results.extend(self.generate_parse_trees(target, new_string, derivation + [step]))
            i += 1

        return results  # Явный возврат списка

    def print_parse_trees(self, target: str):
        """
        Выводит все деревья вывода для заданной цепочки в консоль и лог-файл.
        """
        trees = self.generate_parse_trees(target)
        if not trees:
            print(f"Цепочка '{target}' не может быть выведена по данной грамматикой.")
            return

        print(f"\nВсе возможные деревья вывода для цепочки '{target}':")
        for i, tree in enumerate(trees, 1):
            print(f"\nДерево {i}:")
            for step in tree:
                print(step)


# Запускаем логирование
logger = FileLogger("parse_trees.log")
sys.stdout = logger

try:
    # Пример использования для данной грамматики
    rules = {
        'S': ['aSbS', 'bSaS', '']
    }
    terminals = {'a', 'b'}
    nonterminals = {'S'}

    parser = CFGParser(rules, nonterminals, terminals)

    # Увеличиваем лимит рекурсии (временная мера, если нужно)
    sys.setrecursionlimit(2000)

    print("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА")
    print("=" * 60)
    target_string = "abab"
    parser.print_parse_trees(target_string)

    print("\n=== ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ ===")
    print("=" * 60)

finally:
    # Закрываем логгер и восстанавливаем stdout
    logger.close()