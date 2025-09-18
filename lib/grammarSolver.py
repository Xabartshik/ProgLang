from collections import deque
import random
import re
import datetime
from typing import List, Set, Tuple, Optional

class GrammarSolver:
    """
    Класс для решения задач по формальным грамматикам.
    Правила, алфавиты терминальных и нетерминальных символов задаются в конструкторе.

    ОБНОВЛЕННАЯ ВЕРСИЯ - улучшены методы generate_description и from_language_description
    """

    def __init__(self, rules: dict[str, list[str]], nonterminals: set[str], terminals: set[str],
                 start_symbol: str = 'S'):
        """
        :param rules: Словарь продукций: ключ - левая часть (str), значение - список правых частей (str).
        :param nonterminals: Множество нетерминальных символов.
        :param terminals: Множество терминальных символов.
        :param start_symbol: Стартовый символ (по умолчанию 'S').
        """
        self.rules = rules
        self.nonterminals = nonterminals
        self.terminals = terminals
        self.start = start_symbol

        # Предварительно сортируем правила по длине левой части (длинные первыми)
        self._sorted_rules = sorted(self.rules.items(), key=lambda x: len(x[0]), reverse=True)

    @classmethod
    def from_language_description(cls, terminals: set[str], language_desc: str, start_symbol: str = 'S'):
        """
        Универсальный метод для создания грамматики на основе описания языка и терминалов.

        :param terminals: Множество терминальных символов для создания грамматики
        :param language_desc: Описание языка в формате L = {...}
        :param start_symbol: Стартовый символ грамматики
        :return: Экземпляр GrammarSolver
        :raises ValueError: Если терминалы не заданы
        """
        if not terminals:
            raise ValueError("Ошибка: терминалы не заданы! Для создания грамматики необходимо указать множество терминалов.")

        # Очищаем описание от лишних символов
        clean_desc = language_desc.strip().replace('L = ', '').replace('{', '').replace('}', '').lower()

        # Сортируем терминалы для детерминированности
        sorted_terminals = sorted(terminals - {'ε'})

        # Анализируем описание для определения типа грамматики

        # 1. Паттерн a^n b^n ... (контекстно-свободная грамматика)
        if ('n' in clean_desc and len(sorted_terminals) >= 2 and
            any(f'{t}^n' in clean_desc for t in sorted_terminals)):

            # Проверяем, одинаковые ли показатели степени
            same_exponents = True
            for i, t in enumerate(sorted_terminals[1:], 1):
                if f'{t}^n' not in clean_desc:
                    same_exponents = False
                    break

            if same_exponents and len(sorted_terminals) == 2:
                # Паттерн a^n b^n
                a, b = sorted_terminals[0], sorted_terminals[1]
                rules = {start_symbol: [f'{a}{start_symbol}{b}', 'ε']}
                nonterminals = {start_symbol}

            elif same_exponents and len(sorted_terminals) == 3:
                # Паттерн a^n b^n c^n (контекстно-зависимая)
                a, b, c = sorted_terminals[0], sorted_terminals[1], sorted_terminals[2]
                rules = {
                    start_symbol: [f'{a}{start_symbol}{b}{c}', f'{a}{b}{c}']
                }
                nonterminals = {start_symbol}
            else:
                # Общий случай для множественных терминалов
                rules = {start_symbol: []}
                for t in sorted_terminals:
                    rules[start_symbol].append(f'{t}{start_symbol}')
                rules[start_symbol].append('ε')
                nonterminals = {start_symbol}

        # 2. Паттерн a^n b^m ... (регулярная грамматика с разными показателями)
        elif (any(f'{t}^n_{t}' in clean_desc or f'{t}^m' in clean_desc or f'{t}^k' in clean_desc
                  for t in sorted_terminals)):

            # Создаем цепочку нетерминалов для каждого терминала
            nonterminals = {start_symbol}
            current_nt = start_symbol
            rules = {}

            for i, terminal in enumerate(sorted_terminals):
                if i == len(sorted_terminals) - 1:
                    # Последний терминал
                    rules[current_nt] = [f'{terminal}{current_nt}', terminal]
                else:
                    # Промежуточные терминалы
                    next_nt = chr(ord('A') + i + 1) if chr(ord('A') + i + 1) != start_symbol else chr(ord('A') + i + 2)
                    nonterminals.add(next_nt)
                    rules[current_nt] = [f'{terminal}{current_nt}', next_nt]
                    current_nt = next_nt

        # 3. Паттерн группировки типа 0^n(10)^m
        elif ('(' in clean_desc and ')' in clean_desc and len(sorted_terminals) >= 2):
            # Ищем группы в скобках
            group_pattern = r'\(([^)]+)\)'
            groups = re.findall(group_pattern, clean_desc)

            if groups and len(sorted_terminals) >= 2:
                group_content = groups[0]
                # Создаем грамматику для группировки
                nonterminals = {start_symbol, 'A', 'B'}
                first_terminal = sorted_terminals[0]

                rules = {
                    start_symbol: ['A'],
                    'A': [f'{first_terminal}A', 'B'],
                    'B': [f'{group_content}B', 'ε']
                }

        # 4. Паттерн палиндрома
        elif ('палиндром' in clean_desc or 'w^r' in clean_desc or 'w = w^r' in clean_desc):
            nonterminals = {start_symbol}
            rules = {start_symbol: ['ε']}

            # Добавляем правила для каждого терминала
            for terminal in sorted_terminals:
                rules[start_symbol].extend([terminal, f'{terminal}{start_symbol}{terminal}'])

        # 5. Общий случай - регулярная грамматика
        else:
            nonterminals = {start_symbol}
            rules = {start_symbol: list(sorted_terminals) + ['ε']}

            # Если больше одного терминала, добавляем рекурсию
            if len(sorted_terminals) > 1:
                for terminal in sorted_terminals:
                    rules[start_symbol].append(f'{terminal}{start_symbol}')

        return cls(rules, nonterminals, terminals, start_symbol)

    def generate_description(self) -> str:
        """
        ПЕРЕПИСАННЫЙ метод генерации описания языка.
        Использует терминалы класса для создания описания с переменными n_t1, n_t2, ...
        """
        # Получаем отсортированные терминалы (исключая ε)
        clean_terminals = sorted(self.terminals - {'ε'})

        if not clean_terminals:
            return "ε"

        # Анализируем структуру правил
        grammar_type = self.get_grammar_type()

        # 1. Анализ контекстно-свободных правил (тип 2)
        if grammar_type == 2:
            # Ищем паттерны вида S -> aSb, S -> aSbSc и т.д.
            for left, rights in self.rules.items():
                for right in rights:
                    if left in right:
                        # Анализируем структуру правила
                        left_pos = right.find(left)
                        prefix = right[:left_pos]
                        suffix = right[left_pos + len(left):]

                        # Собираем терминалы в префиксе и суффиксе
                        prefix_terminals = [c for c in prefix if c in clean_terminals]
                        suffix_terminals = [c for c in suffix if c in clean_terminals]

                        if len(prefix_terminals) == len(suffix_terminals) and prefix_terminals and suffix_terminals:
                            # Паттерн вида a^n b^n
                            if len(prefix_terminals) == 1:
                                t1, t2 = prefix_terminals[0], suffix_terminals[0]
                                return f"{t1}^n_{t1} {t2}^n_{t2} | n_{t1} ≥ 0"
                            else:
                                # Множественные символы
                                prefix_str = ''.join(prefix_terminals)
                                suffix_str = ''.join(suffix_terminals)
                                return f"{prefix_str}^n {suffix_str}^n | n ≥ 0"

                        # Проверяем палиндромы
                        if prefix_terminals == suffix_terminals[::-1]:
                            terms_desc = ', '.join(clean_terminals)
                            return f"w | w = w^R, w ∈ {{{terms_desc}}}*"

        # 2. Анализ регулярных правил (тип 3)
        if grammar_type == 3:
            # Создаем описание с независимыми переменными для каждого терминала
            if len(clean_terminals) == 1:
                t = clean_terminals[0]
                return f"{t}^n_{t} | n_{t} ≥ 0"
            elif len(clean_terminals) == 2:
                t1, t2 = clean_terminals[0], clean_terminals[1]
                return f"{t1}^n_{t1} {t2}^n_{t2} | n_{t1}, n_{t2} ≥ 0"
            elif len(clean_terminals) == 3:
                t1, t2, t3 = clean_terminals[0], clean_terminals[1], clean_terminals[2]
                return f"{t1}^n_{t1} {t2}^n_{t2} {t3}^n_{t3} | n_{t1}, n_{t2}, n_{t3} ≥ 0"
            else:
                # Общий случай для произвольного количества терминалов
                terms_with_vars = []
                var_conditions = []

                for i, t in enumerate(clean_terminals):
                    terms_with_vars.append(f"{t}^n_{t}")
                    var_conditions.append(f"n_{t} ≥ 0")

                terms_str = ' '.join(terms_with_vars)
                conditions_str = ', '.join(var_conditions)
                return f"{terms_str} | {conditions_str}"

        # 3. Анализ контекстно-зависимых правил (тип 1)
        if grammar_type <= 1:
            # Для контекстно-зависимых грамматик также используем независимые переменные
            if len(clean_terminals) <= 3:
                terms_with_vars = []
                var_conditions = []

                for i, t in enumerate(clean_terminals):
                    terms_with_vars.append(f"{t}^n_{t}")
                    var_conditions.append(f"n_{t} ≥ 0")

                terms_str = ' '.join(terms_with_vars)
                conditions_str = ', '.join(var_conditions)
                return f"{terms_str} | {conditions_str}"

        # 4. Поиск групповых паттернов типа 0^n(10)^m
        for left, rights in self.rules.items():
            for right in rights:
                # Ищем повторяющиеся последовательности
                if len(clean_terminals) >= 2:
                    t1, t2 = clean_terminals[0], clean_terminals[1]
                    sequence = f"{t1}{t2}"
                    if sequence in right:
                        return f"{t1}^n_{t1}({sequence})^n_{sequence} | n_{t1} ≥ 0, n_{sequence} ≥ 0"

        # 5. Общий случай - все терминалы с независимыми переменными
        terms_with_vars = []
        var_conditions = []

        for t in clean_terminals:
            terms_with_vars.append(f"{t}^n_{t}")
            var_conditions.append(f"n_{t} ≥ 0")

        if terms_with_vars:
            terms_str = ' '.join(terms_with_vars)
            conditions_str = ', '.join(var_conditions)
            return f"{terms_str} | {conditions_str}"

        return "ε"

    def export_to_file(self, filename: str, include_generation: bool = True):
        """
        Экспортирует полный анализ грамматики в текстовый файл.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("АНАЛИЗ ФОРМАЛЬНОЙ ГРАММАТИКИ\n")
            f.write("=" * 60 + "\n")
            f.write(f"Дата создания: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Основная информация о грамматике
            f.write("1. ОПРЕДЕЛЕНИЕ ГРАММАТИКИ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Стартовый символ: {self.start}\n")
            f.write(f"Нетерминалы: {sorted(self.nonterminals)}\n")
            f.write(f"Терминалы: {sorted(self.terminals)}\n\n")

            f.write("Правила грамматики:\n")
            for left, rights in self.rules.items():
                rights_str = " | ".join(r if r != 'ε' else 'ε' for r in rights)
                f.write(f"  {left} -> {rights_str}\n")
            f.write("\n")

            # Классификация
            f.write("2. КЛАССИФИКАЦИЯ ПО ХОМСКОМУ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Тип грамматики: {self.get_grammar_type_name()}\n")
            f.write(f"Номер типа: {self.get_grammar_type()}\n\n")

            # Описание языка
            f.write("3. ОПИСАНИЕ ЯЗЫКА\n")
            f.write("-" * 30 + "\n")
            desc = self.generate_description()
            f.write(f"L(G) = {{{desc}}}\n\n")

            # Статистика
            f.write("4. СТАТИСТИКА\n")
            f.write("-" * 30 + "\n")
            stats = self.get_statistics()
            for key, value in stats.items():
                f.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
            f.write("\n")

            # Генерация строк (если включена)
            if include_generation:
                f.write("5. ПРИМЕРЫ СТРОК ЯЗЫКА\n")
                f.write("-" * 30 + "\n")

                try:
                    strings = self.generate_language_with_all_terminals(method='exhaustive', max_length=20, max_strings=20)
                    if strings:
                        f.write("Строки языка (до 20 символов):\n")
                        for i, string in enumerate(strings[:20], 1):
                            display = string if string else 'ε'
                            f.write(f"  {i:2d}. '{display}' (длина: {len(string)})\n")
                    else:
                        f.write("Строки языка не найдены или язык пустой.\n")
                except Exception as e:
                    f.write(f"Ошибка при генерации строк: {str(e)}\n")

                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("КОНЕЦ АНАЛИЗА\n")
            f.write("=" * 60 + "\n")

    def derive_string(self, target: str) -> list[str]:
        """
        Построение вывода для заданной цепочки с использованием обхода в ширину.
        """
        from collections import deque
        queue = deque([(self.start, [self.start])])
        visited = set([self.start])
        target_symbols = set(target)
        non_terminal_symbols = self.terminals - target_symbols
        if non_terminal_symbols:
            return [
                f"Ошибка ЯЗЫК НЕ ТОТ: в целевой цепочке '{target}' отсутствуют символы из алфавита терминалов: {sorted(non_terminal_symbols)}"]

        max_iterations = 50000
        iteration = 0

        while queue and iteration < max_iterations:
            iteration += 1
            current, path = queue.popleft()
            if current == target:
                return path

            for left, rights in self._sorted_rules:
                for right in rights:
                    pos = current.find(left)
                    if pos != -1:
                        replacement = '' if right == 'ε' else right
                        new_current = current[:pos] + replacement + current[pos + len(left):]
                        if new_current not in visited:
                            visited.add(new_current)
                            queue.append((new_current, path + [f"{left} -> {right}: {new_current}"]))

        return ["Вывод не найден"]

    def get_grammar_type(self) -> int:
        if not self.rules:
            return 0

        def is_right_linear():
            for left, rights in self.rules.items():
                if len(left) != 1 or left not in self.nonterminals:
                    return False
                for right in rights:
                    if right == '' or right == 'ε':
                        continue
                    terminals_part = right[:-1] if right[-1] in self.nonterminals else right
                    if any(c in self.nonterminals for c in terminals_part):
                        return False
            return True

        def is_left_linear():
            for left, rights in self.rules.items():
                if len(left) != 1 or left not in self.nonterminals:
                    return False
                for right in rights:
                    if right == '' or right == 'ε':
                        continue
                    terminals_part = right[1:] if right[0] in self.nonterminals else right
                    if any(c in self.nonterminals for c in terminals_part):
                        return False
            return True

        if is_right_linear() or is_left_linear():
            return 3

        is_context_free = all(len(left) == 1 and left in self.nonterminals for left in self.rules)
        if is_context_free:
            return 2

        is_context_sensitive = all(
            len(left) > 0 and all(
                (right == '' or right == 'ε') and left == self.start or
                (right != '' and right != 'ε' and len(right) >= len(left))
                for right in rights
            )
            for left, rights in self.rules.items()
        )
        if is_context_sensitive:
            return 1

        return 0

    def generate_language(self, method: str = 'exhaustive', max_length: int = 10,
                          max_strings: int = 100, max_depth: int = 15) -> List[str]:
        """
        Генерирует строки языка, порождаемого грамматикой.
        """
        if method == 'exhaustive':
            return self._generate_language_exhaustive(max_length, max_strings)
        elif method == 'random':
            return self._generate_language_random(max_strings, max_depth, max_length)
        else:
            raise ValueError("Метод должен быть 'exhaustive' или 'random'")

    def _generate_language_exhaustive(self, max_length: int, max_strings: int) -> List[str]:
        """
        Генерирует все возможные строки языка до определенной длины методом поиска в ширину.
        """
        queue = deque([self.start])
        result = set()
        processed = set()
        max_iterations = 50000

        iteration = 0
        while queue and len(result) < max_strings and iteration < max_iterations:
            iteration += 1
            current_form = queue.popleft()

            if current_form in processed:
                continue
            processed.add(current_form)

            if self._contains_only_terminals(current_form):
                if len(current_form) <= max_length:
                    result.add(current_form)
                continue

            if len(current_form) > max_length * 5:
                continue

            new_forms = self._apply_all_rules(current_form)
            for new_form in new_forms:
                if new_form not in processed and len(new_form) <= max_length * 5:
                    queue.append(new_form)

        return sorted(result, key=lambda x: (len(x), x))

    def _contains_only_terminals(self, string: str) -> bool:
        """
        Проверяет, содержит ли строка только терминальные символы.
        """
        if not string:
            return True

        for nonterminal in sorted(self.nonterminals, key=len, reverse=True):
            if nonterminal in string:
                return False

        return all(char in self.terminals for char in string)

    def _apply_all_rules(self, current_form: str) -> List[str]:
        """
        Применяет все возможные правила к текущей форме.
        """
        new_forms = []

        for left, rights in self._sorted_rules:
            positions = self._find_all_positions(current_form, left)

            for pos in positions:
                for right in rights:
                    replacement = '' if right == 'ε' else right
                    new_form = current_form[:pos] + replacement + current_form[pos + len(left):]
                    new_forms.append(new_form)

        return new_forms

    def _find_all_positions(self, text: str, pattern: str) -> List[int]:
        """Находит все позиции вхождения pattern в text."""
        positions = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def _generate_language_random(self, count: int, max_depth: int, max_length: int) -> List[str]:
        """
        Генерирует случайные строки языка.
        """
        result = set()
        attempts = 0
        max_attempts = count * 500

        while len(result) < count and attempts < max_attempts:
            attempts += 1
            try:
                string = self._generate_random_derivation(self.start, max_depth, max_length)
                if string is not None and len(string) <= max_length:
                    result.add(string)
            except (RecursionError, Exception):
                continue

        return sorted(result, key=lambda x: (len(x), x))

    def _generate_random_derivation(self, current_form: str, max_depth: int, max_length: int) -> Optional[str]:
        """
        Рекурсивная генерация случайной строки с применением правил.
        """
        if max_depth <= 0:
            return current_form if self._contains_only_terminals(current_form) else None

        if len(current_form) > max_length * 4:
            return None

        if self._contains_only_terminals(current_form):
            return current_form

        applicable_rules = []

        for left, rights in self._sorted_rules:
            positions = self._find_all_positions(current_form, left)
            for pos in positions:
                for right in rights:
                    applicable_rules.append((left, right, pos))

        if not applicable_rules:
            return current_form if self._contains_only_terminals(current_form) else None

        left, right, pos = random.choice(applicable_rules)
        replacement = '' if right == 'ε' else right
        new_form = current_form[:pos] + replacement + current_form[pos + len(left):]

        return self._generate_random_derivation(new_form, max_depth - 1, max_length)

    def format_language_output(self, grammar_name: str, language_description: str,
                               strings: List[str]) -> str:
        """
        Форматирует вывод языка в стиле L(G) = {...}.
        """
        output = f"L({grammar_name}) = {{{language_description}}}\n"
        if not strings:
            output += "  ∅ (пустой язык)\n"
        else:
            for i, string in enumerate(strings, 1):
                display_string = string if string else 'ε'
                output += f"  {i:2d}. {display_string}\n"
        return output

    def print_grammar(self):
        """
        Выводит грамматику в читаемом виде.
        """
        print(f"Стартовый символ: {self.start}")
        print(f"Нетерминалы: {sorted(self.nonterminals)}")
        print(f"Терминалы: {sorted(self.terminals)}")
        print("Правила грамматики:")
        for left, rights in self.rules.items():
            rights_str = " | ".join(r if r != 'ε' else 'ε' for r in rights)
            print(f"  {left} -> {rights_str}")
        print()

    def get_grammar_type_name(self) -> str:
        """
        Возвращает название типа грамматики по классификации Хомского.
        """
        type_num = self.get_grammar_type()
        type_names = {
            0: "Неограниченная (тип 0)",
            1: "Контекстно-зависимая (тип 1)",
            2: "Контекстно-свободная (тип 2)",
            3: "Регулярная (тип 3)"
        }
        return type_names.get(type_num, "Неизвестный тип")

    def has_all_terminals(self, string: str) -> bool:
        """
        Проверяет, содержит ли цепочка все терминальные символы из алфавита.
        """
        string_set = set(string)
        clean_terminals = self.terminals - {'ε'}
        return all(t in string_set for t in clean_terminals)

    def generate_language_with_all_terminals(self, method: str = 'exhaustive', max_length: int = 10,
                                             max_strings: int = 100, max_depth: int = 15) -> List[str]:
        """
        Генерирует строки языка, содержащие все терминальные символы.
        """
        all_strings = self.generate_language(method=method, max_length=max_length,
                                             max_strings=max_strings * 20, max_depth=max_depth)
        filtered_strings = [s for s in all_strings if self.has_all_terminals(s)]
        return sorted(filtered_strings[:max_strings], key=lambda x: (len(x), x))

    def get_statistics(self) -> dict:
        """
        Возвращает статистику грамматики.
        """
        return {
            'правила': len(self.rules),
            'нетерминалы': len(self.nonterminals),
            'терминалы': len(self.terminals),
            'тип': self.get_grammar_type(),
            'тип_название': self.get_grammar_type_name(),
            'многосимвольные_левые_части': sum(1 for left in self.rules.keys() if len(left) > 1),
            'правила_с_эпсилон': sum(1 for rights in self.rules.values() for right in rights if right == 'ε')
        }
