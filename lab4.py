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

        # Проверяем, возможна ли целевая строка. Условие оптимизации - если строка не является префиксом,
        # получить из нее вряд-ли что-то получится
        if not self.is_possible_prefix(current_string, target):
            return []

        # Проверяем, не слишком ли много терминалов. Условие оптимизации.
        # Отсекает продолжение изучения
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

    def is_right_linear(self) -> bool:
        """
        Проверяет, является ли грамматика праволинейной.
        :return: True, если грамматика праволинейная, иначе False.
        """
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':  # ε-правило допустимо
                    continue
                # Проверяем, что правило вида a или aB (a - терминал, B - нетерминал или пусто)
                if len(prod) > 2 or (len(prod) == 2 and (prod[0] not in self.terminals or prod[1] not in self.nonterminals)):
                    return False
                if len(prod) == 1 and prod[0] not in self.terminals:
                    return False
        return True

    def to_left_linear_grammar(self) -> Dict[str, List[str]]:
        """
        Преобразует праволинейную грамматику в эквивалентную леволинейную, допускающую детерминированный разбор.
        :return: Словарь правил леволинейной грамматики.
        """
        if not self.is_right_linear():
            print("Грамматика не является праволинейной. Преобразование невозможно.")
            return {}

        # Шаг 1: Построить НКА из праволинейной грамматики
        nfa_states = self.nonterminals | {'F'}  # Добавляем финальное состояние F
        nfa_transitions = {state: {} for state in nfa_states}
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':  # ε-переход в финальное состояние
                    nfa_transitions[nonterminal].setdefault('', set()).add('F')
                elif len(prod) == 1:  # a -> F
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add('F')
                else:  # aB
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add(prod[1])

        # Шаг 2: Преобразование НКА в ДКА
        dfa_states = set()
        dfa_transitions = {}
        state_queue = [frozenset([self.start_symbol])]
        dfa_states.add(frozenset([self.start_symbol]))
        final_states = set()

        while state_queue:
            current_state_set = state_queue.pop(0)
            dfa_transitions[current_state_set] = {}

            # Для каждого терминала проверяем переходы
            for terminal in self.terminals:
                next_states = set()
                for state in current_state_set:
                    if terminal in nfa_transitions[state]:
                        next_states.update(nfa_transitions[state][terminal])
                if next_states:
                    next_state_set = frozenset(next_states)
                    if next_state_set not in dfa_states:
                        dfa_states.add(next_state_set)
                        state_queue.append(next_state_set)
                    dfa_transitions[current_state_set][terminal] = next_state_set

            # Проверяем ε-переходы (в финальное состояние)
            if 'F' in current_state_set:
                final_states.add(current_state_set)

        # Шаг 3: Построить леволинейную грамматику из ДКА
        left_linear_rules = {}
        state_to_nonterminal = {state: f"Q{index}" for index, state in enumerate(dfa_states)}
        state_to_nonterminal[frozenset([self.start_symbol])] = self.start_symbol

        for state in dfa_states:
            nonterminal = state_to_nonterminal[state]
            left_linear_rules[nonterminal] = []
            # Добавляем правила для переходов
            if state in dfa_transitions:
                for terminal, next_state in dfa_transitions[state].items():
                    next_nonterminal = state_to_nonterminal[next_state]
                    left_linear_rules[nonterminal].append(f"{next_nonterminal}{terminal}")
            # Если состояние финальное, добавляем ε-продукцию
            if state in final_states:
                left_linear_rules[nonterminal].append('')

        return left_linear_rules

    def print_grammar(self):
        """
        Выводит грамматику в читаемом виде.
        """
        print(f"Стартовый символ: {self.start_symbol}")
        print(f"Нетерминалы: {sorted(self.nonterminals)}")
        print(f"Терминалы: {sorted(self.terminals)}")
        print("Правила грамматики:")
        for left, rights in self.rules.items():
            rights_str = " | ".join(r if r != 'ε' else 'ε' for r in rights)
            print(f"  {left} -> {rights_str}")
        print()

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

    left_linear_b = parser_b.to_left_linear_grammar()
    parser_b.print_grammar()

    # Тестируем деревья вывода для цепочки
    target_string_b = "ab$"
    parser_b.print_parse_trees(target_string_b)

    print("\n=== ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ ===")
    print("=" * 60)

finally:
    # Закрываем логгер и восстанавливаем stdout
    logger.close()