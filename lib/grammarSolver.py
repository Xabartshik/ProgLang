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

    def __init__(self, rules: dict[str, list[str]], nonterminals=None, terminals=None,
                 start_symbol: str = 'S'):
        """
        :param rules: Словарь продукций: ключ - левая часть (str), значение - список правых частей (str).
        :param nonterminals: Множество нетерминальных символов (необязательно - будет определено автоматически).
        :param terminals: Множество терминальных символов (необязательно - будет определено автоматически).
        :param start_symbol: Стартовый символ.
        """

        self.rules = rules
        self.start = start_symbol

        # Если терминалы и нетерминалы не заданы, определяем их автоматически
        if nonterminals is None or terminals is None:
            auto_nonterminals, auto_terminals = self._auto_detect_symbols()
            self.nonterminals = nonterminals if nonterminals is not None else auto_nonterminals
            self.terminals = terminals if terminals is not None else auto_terminals
        else:
            self.nonterminals = set(nonterminals) if nonterminals else set()
            self.terminals = set(terminals) if terminals else set()

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
            raise ValueError(
                "Ошибка: терминалы не заданы! Для создания грамматики необходимо указать множество терминалов.")

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
                    strings = self.generate_language_with_all_terminals(method='exhaustive', max_length=20,
                                                                        max_strings=20)
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

        # Проверяем, что целевая строка состоит только из терминалов
        if not self._can_decompose_to_terminals(target):
            return [f"Ошибка: строка '{target}' не может быть разложена на терминалы грамматики"]

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

    def generate_language(self, method: str = 'cycle_detection', max_length: int = 20,
                          max_strings: int = 100, max_depth: int = 15,
                          require_all_terminals: bool = False) -> List[str]:
        """
        Генерирует цепочки языка с защитой от циклов.

        :param method: Метод генерации ('cycle_detection', 'depth_limited', 'length_limited',
                       'exhaustive', 'random', 'breadth_first', 'hybrid')
        :param max_length: Максимальная длина генерируемых строк
        :param max_strings: Максимальное количество строк для генерации
        :param max_depth: Максимальная глубина вывода
        :param require_all_terminals: Если True, возвращает только строки, содержащие все терминалы
        :return: Список сгенерированных строк
        """

        # Выбор метода генерации
        if method == 'cycle_detection':
            all_strings = self._generate_cycle_detection(max_strings * 3, max_length, max_depth)
        elif method == 'depth_limited':
            all_strings = self._generate_depth_limited(max_depth, max_strings * 3, max_length)
        elif method == 'length_limited':
            all_strings = self._generate_length_limited(max_length, max_strings * 3)
        elif method == 'breadth_first':
            all_strings = self._generate_breadth_first(max_depth, max_strings * 3, max_length)
        elif method == 'hybrid':
            all_strings = self._generate_hybrid(max_depth, max_length, max_strings * 3)
        elif method == 'exhaustive':
            all_strings = self._generate_language_exhaustive_safe(max_length, max_strings * 3)
        elif method == 'random':
            all_strings = self._generate_language_random_safe(max_strings * 3, max_depth, max_length)
        else:
            # По умолчанию используем cycle_detection как самый безопасный
            all_strings = self._generate_cycle_detection(max_strings * 3, max_length, max_depth)

        # Фильтрация по требованию всех терминалов
        if require_all_terminals:
            filtered_strings = []
            for string in all_strings:
                if self.has_all_terminals(string) and len(filtered_strings) < max_strings:
                    filtered_strings.append(string)
            return sorted(filtered_strings, key=lambda x: (len(x), x))[:max_strings]

        return sorted(all_strings, key=lambda x: (len(x), x))[:max_strings]

    def _generate_recursive(self, max_strings: int = 100,
                            max_length: int = 20,
                            max_depth: int = 10) -> List[str]:
        """
        Рекурсивная генерация всех возможных цепочек строк для заданной грамматики,
        с фильтрацией по длине и требованию наличия всех терминалов.

        :param max_strings: Максимальное количество строк для генерации
        :param max_length: Максимальная длина генерируемых строк
        :param max_depth: Максимальная глубина рекурсии для предотвращения бесконечных циклов
        :return: Список строк, каждая из которых содержит все терминалы языка
        """

        def derive_recursive(symbol: str, depth: int = 0) -> List[str]:
            if depth > max_depth:
                return []
            if symbol not in self.rules:
                return [symbol]

            result = []
            for production in self.rules[symbol]:
                current_strings = [""]
                for char in production:
                    sub_strings = derive_recursive(char, depth + 1)
                    new_strings = []
                    for curr in current_strings:
                        for sub in sub_strings:
                            new_strings.append(curr + sub)
                    current_strings = new_strings
                result.extend(current_strings)
            return result

        # Генерируем все возможные строки от стартового символа
        all_strings = derive_recursive(self.start)

        # Фильтруем по длине и наличию всех терминалов (исключая ε)
        clean_terminals = self.terminals - {'ε'}
        filtered = []
        for s in all_strings:
            if len(s) <= max_length and self.has_all_terminals(s):
                filtered.append(s)

        # Сортируем и обрезаем до max_strings
        filtered.sort(key=lambda x: (len(x), x))
        return filtered[:max_strings]

    def _generate_cycle_detection(self, max_strings: int = 100, max_length: int = 20,
                                  max_depth: int = 15) -> List[str]:
        """Генерация с обнаружением и избеганием циклов"""
        generated = set()
        visited_states = set()
        queue = [(self.start, 0)]  # (строка, глубина)

        while queue and len(generated) < max_strings:
            current, depth = queue.pop(0)

            # Множественная защита
            if (depth > max_depth or
                    len(current) > max_length or
                    current in visited_states):
                continue

            visited_states.add(current)

            # Если строка терминальная, добавляем в результат
            if self._contains_only_terminals(current):
                generated.add(current)
                continue

            # Расширение первого нетерминала
            expanded = False
            for i, symbol in enumerate(current):
                if symbol in self.nonterminals and symbol in self.rules:
                    for rule in self.rules[symbol]:
                        new_string = current[:i] + rule + current[i + 1:]
                        if new_string not in visited_states:
                            queue.append((new_string, depth + 1))
                    expanded = True
                    break

            if not expanded:
                continue

        return list(generated)

    def _generate_depth_limited(self, max_depth: int = 10, max_strings: int = 100,
                                max_length: int = 20) -> List[str]:
        """Генерация с ограничением глубины"""
        generated = set()

        def derive_recursive(current: str, depth: int):
            if depth >= max_depth or len(generated) >= max_strings or len(current) > max_length:
                return

            if self._contains_only_terminals(current):
                generated.add(current)
                return

            # Найти первый нетерминал
            for i, symbol in enumerate(current):
                if symbol in self.nonterminals and symbol in self.rules:
                    for rule in self.rules[symbol]:
                        new_string = current[:i] + rule + current[i + 1:]
                        derive_recursive(new_string, depth + 1)
                    break

        derive_recursive(self.start, 0)
        return list(generated)

    def _generate_length_limited(self, max_length: int = 20, max_strings: int = 100) -> List[str]:
        """Генерация с ограничением длины строки"""
        generated = set()
        queue = [self.start]

        while queue and len(generated) < max_strings:
            current = queue.pop(0)

            if len(current) > max_length:
                continue

            if self._contains_only_terminals(current):
                generated.add(current)
                continue

            # Расширяем первый нетерминал
            for i, symbol in enumerate(current):
                if symbol in self.nonterminals and symbol in self.rules:
                    for rule in self.rules[symbol]:
                        new_string = current[:i] + rule + current[i + 1:]
                        if len(new_string) <= max_length:
                            queue.append(new_string)
                    break

        return list(generated)

    def _generate_breadth_first(self, max_level: int = 5, max_strings: int = 100,
                                max_length: int = 20) -> List[str]:
        """Генерация по уровням с ограничением"""
        generated = set()
        current_level = [self.start]

        for level in range(max_level):
            if len(generated) >= max_strings:
                break

            next_level = []

            for string in current_level:
                if len(string) > max_length:
                    continue

                if self._contains_only_terminals(string):
                    generated.add(string)
                    continue

                # Расширить первый нетерминал
                for i, symbol in enumerate(string):
                    if symbol in self.nonterminals and symbol in self.rules:
                        for rule in self.rules[symbol]:
                            new_string = string[:i] + rule + string[i + 1:]
                            next_level.append(new_string)
                        break

            current_level = list(set(next_level))  # Убрать дубликаты

        return list(generated)

    def _generate_hybrid(self, max_depth: int = 8, max_length: int = 15,
                         max_strings: int = 100) -> List[str]:
        """Гибридный подход с несколькими ограничениями"""
        generated = set()
        visited = set()

        def derive_safe(current: str, depth: int):
            if (depth >= max_depth or
                    len(current) > max_length or
                    current in visited or
                    len(generated) >= max_strings):
                return

            visited.add(current)

            if self._contains_only_terminals(current):
                generated.add(current)
                return

            for i, symbol in enumerate(current):
                if symbol in self.nonterminals and symbol in self.rules:
                    for rule in self.rules[symbol]:
                        new_string = current[:i] + rule + current[i + 1:]
                        derive_safe(new_string, depth + 1)
                    break

        derive_safe(self.start, 0)
        return list(generated)

    def _generate_language_exhaustive_safe(self, max_length: int, max_strings: int) -> List[str]:
        """Безопасная версия исчерпывающей генерации"""
        from collections import deque

        queue = deque([self.start])
        result = set()
        processed = set()
        max_iterations = 50000
        iteration = 0

        while queue and len(result) < max_strings and iteration < max_iterations:
            iteration += 1
            current_form = queue.popleft()

            if current_form in processed or len(current_form) > max_length:
                continue

            processed.add(current_form)

            if self._contains_only_terminals(current_form):
                result.add(current_form)
                continue

            expanded = False
            for left, rights in self.rules.items():
                positions = self._find_all_positions(current_form, left)
                for pos in positions:
                    for right in rights:
                        replacement = right if right != 'ε' else ''
                        new_form = current_form[:pos] + replacement + current_form[pos + len(left):]

                        if new_form not in processed and len(new_form) <= max_length:
                            queue.append(new_form)
                            expanded = True

            if not expanded and not self._contains_only_terminals(current_form):
                continue

        return list(result)

    def _generate_language_random_safe(self, count: int, max_depth: int, max_length: int) -> List[str]:
        """Безопасная версия случайной генерации"""
        result = set()
        attempts = 0
        max_attempts = count * 500

        while len(result) < count and attempts < max_attempts:
            attempts += 1
            try:
                string = self._generate_random_derivation_safe(self.start, max_depth, max_length)
                if string is not None and len(string) <= max_length:
                    result.add(string)
            except (RecursionError, Exception):
                continue

        return sorted(result, key=lambda x: (len(x), x))

    def _generate_random_derivation_safe(self, current_form: str, max_depth: int, max_length: int) -> Optional[str]:
        """Безопасная версия случайного вывода"""
        if max_depth <= 0 or len(current_form) > max_length:
            return current_form if self._contains_only_terminals(current_form) else None

        if self._contains_only_terminals(current_form):
            return current_form

        # Найти все нетерминалы
        nonterminals_positions = []
        for i, symbol in enumerate(current_form):
            if symbol in self.nonterminals and symbol in self.rules:
                nonterminals_positions.append((i, symbol))

        if not nonterminals_positions:
            return current_form if self._contains_only_terminals(current_form) else None

        # Случайно выбрать нетерминал для замены
        import random
        pos, symbol = random.choice(nonterminals_positions)

        # Случайно выбрать правило
        rules = self.rules.get(symbol, [])
        if rules:
            rule = random.choice(rules)
            replacement = rule if rule != 'ε' else ''
            new_form = current_form[:pos] + replacement + current_form[pos + 1:]
            return self._generate_random_derivation_safe(new_form, max_depth - 1, max_length)

        return None

    def detect_grammar_issues(self) -> List[str]:
        """
        Обнаруживает потенциальные проблемы в грамматике.

        :return: Список описаний обнаруженных проблем
        """
        issues = []

        # 1. Поиск прямых циклов (A → A)
        cycles = []
        for left, rights in self.rules.items():
            for right in rights:
                if right == left:
                    cycles.append(f"{left} → {right}")

        if cycles:
            issues.append(f"Прямые циклы: {cycles}")

        # 2. Поиск взаимных циклов длины 2 (A → B, B → A)
        mutual_cycles = []
        for left1, rights1 in self.rules.items():
            for right1 in rights1:
                if right1 in self.rules:
                    for right2 in self.rules[right1]:
                        if right2 == left1 and left1 != right1:
                            cycle = f"{left1} ↔ {right1}"
                            if cycle not in mutual_cycles:
                                mutual_cycles.append(cycle)

        if mutual_cycles:
            issues.append(f"Взаимные циклы: {mutual_cycles}")

        # 3. Проверка контекстно-зависимых правил (левая часть > 1 символа)
        context_dependent = []
        for left in self.rules.keys():
            if len(left) > 1:
                context_dependent.append(left)

        if context_dependent:
            issues.append(f"Контекстно-зависимые левые части: {context_dependent}")

        # 4. Поиск потенциально растущих правил (A → ...A...)
        growing_rules = []
        for left, rights in self.rules.items():
            for right in rights:
                if len(right) > len(left) and left in right:
                    growing_rules.append(f"{left} → {right}")

        if growing_rules:
            issues.append(f"Потенциально растущие правила: {growing_rules}")

        # 5. Поиск бесплодных нетерминалов (не могут привести к терминалам)
        productive = set(self.terminals)
        changed = True
        while changed:
            changed = False
            for left, rights in self.rules.items():
                if left not in productive:
                    for right in rights:
                        if all(symbol in productive for symbol in right) or right == 'ε':
                            productive.add(left)
                            changed = True
                            break

        unproductive = self.nonterminals - productive
        if unproductive:
            issues.append(f"Бесплодные символы: {sorted(unproductive)}")

        return issues

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
        ИСПРАВЛЕННАЯ версия: правильно работает с многосимвольными терминалами.
        Проверяет, можно ли разложить строку полностью на терминалы.
        """
        if not string:
            return True

        # Сначала проверяем нетерминалы (многосимвольные первыми)
        for nonterminal in sorted(self.nonterminals, key=len, reverse=True):
            if nonterminal in string:
                return False

        # Затем пытаемся разложить строку на терминалы
        return self._can_decompose_to_terminals(string)

    def _can_decompose_to_terminals(self, string: str) -> bool:
        """
        Проверяет, можно ли разложить строку на терминалы используя жадный алгоритм.
        Сначала пытается использовать самые длинные терминалы.
        """
        if not string:
            return True

        # Сортируем терминалы по длине (длинные первыми)
        sorted_terminals = sorted(self.terminals - {'ε'}, key=len, reverse=True)

        # Рекурсивно пытаемся разложить строку
        def decompose(s, pos=0):
            if pos >= len(s):
                return True

            for terminal in sorted_terminals:
                if s[pos:].startswith(terminal):
                    if decompose(s, pos + len(terminal)):
                        return True
            return False

        return decompose(string)

    def _apply_all_rules(self, current_form: str) -> List[str]:
        """
        Применяет все возможные правила к текущей форме слева направо и справа налево.
        Возвращает список всех возможных новых форм.
        """
        new_forms = set()  # Используем set для избежания дубликатов

        # Применение правил слева направо
        for left, rights in self._sorted_rules:
            positions = self._find_all_positions(current_form, left)
            for pos in positions:
                for right in rights:
                    replacement = '' if right == 'ε' else right
                    new_form = current_form[:pos] + replacement + current_form[pos + len(left):]
                    new_forms.add(new_form)

        # Применение правил справа налево
        reversed_form = current_form[::-1]  # Переворачиваем строку
        for left, rights in self._sorted_rules:
            reversed_left = left[::-1]  # Переворачиваем левую часть правила
            positions = self._find_all_positions(reversed_form, reversed_left)
            for pos in positions:
                for right in rights:
                    reversed_right = right[::-1] if right != 'ε' else ''  # Переворачиваем правую часть
                    new_reversed_form = reversed_form[:pos] + reversed_right + reversed_form[pos + len(reversed_left):]
                    new_forms.add(new_reversed_form[::-1])  # Переворачиваем результат обратно

        return list(new_forms)  # Преобразуем set в list для возврата

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

    def _extract_terminals_from_string(self, s: str) -> List[str]:
        """
        Извлекает терминалы из строки, учитывая многосимвольные терминалы.
        """
        if not s:
            return []

        result = []
        pos = 0
        sorted_terminals = sorted(self.terminals - {'ε'}, key=len, reverse=True)

        while pos < len(s):
            found = False
            for terminal in sorted_terminals:
                if s[pos:].startswith(terminal):
                    result.append(terminal)
                    pos += len(terminal)
                    found = True
                    break
            if not found:
                pos += 1  # Пропускаем неизвестный символ

        return result

    def has_all_terminals(self, string: str) -> bool:
        """
        Проверяет, содержит ли цепочка все терминальные символы из алфавита.
        """
        extracted_terminals = set(self._extract_terminals_from_string(string))
        clean_terminals = self.terminals - {'ε'}
        return all(t in extracted_terminals for t in clean_terminals)

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

    def remove_epsilon_rules(self):
        """
        Алгоритм устранения epsilon-правил (ε-правил).

        Шаги:
        1. Найти все ε-порождающие нетерминалы
        2. Создать новые правила без ε-символов
        3. Удалить все ε-правила
        """
        print("Удаление epsilon-правил...")

        # Шаг 1: Найти все ε-порождающие нетерминалы
        epsilon_generating = set()

        # Сначала найдем прямые ε-правила
        for left, rights in self.rules.items():
            if '' in rights or 'ε' in rights:
                epsilon_generating.add(left)

        # Затем найдем косвенные ε-порождающие
        changed = True
        while changed:
            changed = False
            for left, rights in self.rules.items():
                if left not in epsilon_generating:
                    for right in rights:
                        # Если правая часть состоит только из ε-порождающих символов
                        if right and all(
                                symbol in epsilon_generating for symbol in right if symbol in self.nonterminals):
                            epsilon_generating.add(left)
                            changed = True
                            break

        print(f"ε-порождающие нетерминалы: {epsilon_generating}")

        # Шаг 2: Создать новые правила
        new_rules = {}

        for left, rights in self.rules.items():
            new_rules[left] = []

            for right in rights:
                # Пропускаем ε-правила
                if right == '' or right == 'ε':
                    continue

                # Генерируем все возможные варианты правила
                variants = self._generate_epsilon_variants(right, epsilon_generating)

                for variant in variants:
                    if variant not in new_rules[left]:
                        new_rules[left].append(variant)

        # Если стартовый символ может породить ε, добавляем новый стартовый символ
        if self.start in epsilon_generating:
            new_start = self.start + "'"
            while new_start in self.nonterminals:
                new_start += "'"

            new_rules[new_start] = [self.start, '']
            self.nonterminals.add(new_start)
            self.start = new_start

        # Удаляем пустые правила
        self.rules = {left: [r for r in rights if r] for left, rights in new_rules.items() if rights}

        print("Epsilon-правила успешно удалены")
        return self

    def _generate_epsilon_variants(self, rule_right: str, epsilon_generating: set) -> list[str]:
        """Генерирует все варианты правила при удалении ε-порождающих символов"""
        if not rule_right:
            return ['']

        variants = set()

        # Создаем все возможные комбинации присутствия/отсутствия ε-порождающих символов
        def generate_combinations(pos: int, current: str):
            if pos >= len(rule_right):
                if current:  # Не добавляем пустые строки
                    variants.add(current)
                return

            symbol = rule_right[pos]

            # Всегда добавляем символ
            generate_combinations(pos + 1, current + symbol)

            # Если символ ε-порождающий, добавляем вариант без него
            if symbol in epsilon_generating:
                generate_combinations(pos + 1, current)

        generate_combinations(0, '')
        variants.add(rule_right)  # Добавляем оригинальное правило

        return list(variants)

    def remove_unit_productions(self):
        """
        Алгоритм удаления цепных правил (unit productions).

        Цепное правило: A → B, где A и B - нетерминалы

        Шаги:
        1. Найти все цепные пары (A,B)
        2. Для каждой цепной пары создать новые правила
        3. Удалить все цепные правила
        """
        print("Удаление цепных правил...")

        # Шаг 1: Найти все цепные пары
        unit_pairs = set()

        # Добавляем рефлексивные пары (A,A)
        for nt in self.nonterminals:
            unit_pairs.add((nt, nt))

        # Ищем прямые цепные правила и строим транзитивное замыкание
        changed = True
        while changed:
            changed = False
            for left, rights in self.rules.items():
                for right in rights:
                    # Проверяем, является ли правило цепным (A → B)
                    if len(right) == 1 and right in self.nonterminals:
                        if (left, right) not in unit_pairs:
                            unit_pairs.add((left, right))
                            changed = True

                        # Добавляем транзитивные пары
                        for a, b in list(unit_pairs):
                            if b == left and (a, right) not in unit_pairs:
                                unit_pairs.add((a, right))
                                changed = True

        print(f"Цепные пары: {unit_pairs}")

        # Шаг 2: Создать новые правила
        new_rules = {}

        for nt in self.nonterminals:
            new_rules[nt] = []

        # Для каждой цепной пары (A,B) и каждого нецепного правила B → α
        # добавляем правило A → α
        for left, right in unit_pairs:
            if right in self.rules:
                for production in self.rules[right]:
                    # Добавляем только нецепные правила
                    if not (len(production) == 1 and production in self.nonterminals):
                        if production not in new_rules[left]:
                            new_rules[left].append(production)

        # Удаляем пустые правила
        self.rules = {left: rights for left, rights in new_rules.items() if rights}

        print("Цепные правила успешно удалены")
        return self

    def remove_useless_symbols(self):
        """
        Алгоритм удаления бесполезных символов.

        Вход: КС-грамматика G = (VT, VN, P, S).
        Выход: КС-грамматика G' = (VT', VN', P', S), не содержащая бесплодных символов,
               для которой L(G) = L(G').

        Метод:
        • Рекурсивно строим множества N0, N1, ...
        • N0 = ∅, i = 1.
        • Ni = {A | (A → α) ∈ P и α ∈ (Ni-1 ∪ VT)*} ∪ Ni-1.
        • Если Ni ≠ Ni-1, то i = i + 1 и переходим к шагу 2, иначе VN' = Ni;
        • P' состоит из правил множества P, содержащих только символы из VN' ∪ VT;
        • VT' содержит только терминалы, используемые в P'.
        """

        # Шаг 1: Рекурсивное построение множеств N0, N1, ...

        # 0 итерация - пустое множество
        N_prev = set()
        i = 1

        while True:
            # Текущая итерация
            N_current = set(N_prev)  # Ni-1

            # Ищем нетерминалы A, для которых есть правило A → α (из нетерминала можно получить терминал, либо символ с предыдущей итерации),
            # где α состоит только из символов из (Ni-1 ∪ VT)
            productive_symbols = N_prev | self.terminals  # Ni-1 ∪ VT

            for left, rights in self.rules.items():
                if left not in N_current:  # Если A еще не в множестве
                    for right in rights:
                        # Проверяем, состоит ли α только из символов (Ni-1 ∪ VT)
                        if self._is_string_in_alphabet(right, productive_symbols):
                            N_current.add(left)
                            break  # Достаточно одного подходящего правила

            print(f"N{i} = {N_current}")

            # Если текущая и предыдущая итерация не совпадают, то все, то i = i + 1 и переходим к следующей итерации
            if N_current != N_prev:
                N_prev = N_current
                i += 1
            else:
                # Иначе VN' = Ni
                VN_prime = N_current
                break

        # Шаг 2: Построение новых правил P' из правил, содержащих только символы из VN' ∪ VT
        allowed_symbols = VN_prime | self.terminals
        P_prime = {}

        # Неизмененные правила
        for left, rights in self.rules.items():
            if left in VN_prime:  # Левая часть должна быть в VN'
                valid_rights = []
                for right in rights:
                    # Правая часть должна содержать только символы из VN' ∪ VT
                    if self._is_string_in_alphabet(right, allowed_symbols):
                        valid_rights.append(right)

                if valid_rights:  # Добавляем только если есть валидные правила
                    P_prime[left] = valid_rights

        # Шаг 3: Построение VT' - только используемые терминалы
        used_terminals = set()
        for rights in P_prime.values():
            for right in rights:
                for symbol in right:
                    if symbol in self.terminals:
                        used_terminals.add(symbol)
        # Проверяем, что стартовый символ остался продуктивным
        if self.start not in VN_prime:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Стартовый символ {self.start} стал бесплодным!")
            print("Грамматика генерирует пустой язык.")

        # Обновляем грамматику
        removed_nonterminals = self.nonterminals - VN_prime
        removed_terminals = self.terminals - used_terminals

        if removed_nonterminals:
            print(f"Удаленные бесплодные нетерминалы: {removed_nonterminals}")
        if removed_terminals:
            print(f"Удаленные неиспользуемые терминалы: {removed_terminals}")

        self.nonterminals = VN_prime
        self.terminals = used_terminals
        self.rules = P_prime

        print("Бесполезные символы успешно удалены")
        return self

    def _is_string_in_alphabet(self, string: str, alphabet: set) -> bool:
        """Проверяет, состоит ли строка только из символов данного алфавита"""
        if not string:  # Пустая строка (ε) всегда допустима
            return True
        return all(symbol in alphabet for symbol in string)

    def remove_unproductive_symbols(self):
        """
        Удаление бесплодных символов (не могут породить терминальную строку).

        Шаги:
        1. Отметить все терминалы как плодотворные
        2. Итеративно находить плодотворные нетерминалы
        3. Удалить все неплодотворные символы
        """

        # Шаг 1: Все терминалы плодотворны
        productive = set(self.terminals)

        # Шаг 2: Итеративно находим плодотворные нетерминалы
        changed = True
        while changed:
            changed = False
            for left, rights in self.rules.items():
                if left not in productive:
                    for right in rights:
                        # Если правая часть состоит только из плодотворных символов
                        if all(symbol in productive for symbol in right) or right == '':
                            productive.add(left)
                            changed = True
                            break

        # Определяем неплодотворные символы
        unproductive = self.nonterminals - productive
        print(f"Бесплодные символы: {unproductive}")

        # Шаг 3: Удаляем неплодотворные символы
        if unproductive:
            # Обновляем множества символов
            self.nonterminals -= unproductive

            # Удаляем правила с неплодотворными символами
            new_rules = {}
            for left, rights in self.rules.items():
                if left not in unproductive:
                    new_rights = []
                    for right in rights:
                        # Включаем правило только если оно не содержит неплодотворных символов
                        if not any(symbol in unproductive for symbol in right):
                            new_rights.append(right)
                    if new_rights:
                        new_rules[left] = new_rights

            self.rules = new_rules

        return self

    def remove_unreachable_symbols(self):
        """
        Удаление недостижимых символов по алгоритму итеративного расширения множества достижимых:
        1. V0 = {start}
        2. Vi = Vi-1 ∪ { x | есть правило A→α x β, A∈Vi-1 }
        3. Повторять, пока Vi не перестанет изменяться
        4. Оставить только правила и символы из итогового Vi
        """

        # Шаг 1: инициализация начальным символом
        reachable = {self.start}
        changed = True

        # Шаги 2–3: итеративно расширяем множество достижимых символов
        while changed:
            changed = False
            # Для каждой пары (A → production) где A уже достижим
            for A, productions in list(self.rules.items()):
                if A in reachable:
                    for prod in productions:
                        # Для каждого символа x в правиле A→prod
                        for x in prod:
                            if x not in reachable:
                                reachable.add(x)
                                changed = True

        # Итоговое множество достижимых нетерминалов и терминалов
        reachable_nonterminals = reachable & self.nonterminals
        reachable_terminals = reachable & self.terminals

        # Шаг 4: фильтрация грамматики
        # Оставляем только достижимые символы
        self.nonterminals = reachable_nonterminals
        self.terminals = reachable_terminals

        # Оставляем только правила, все символы в которых достижимы
        new_rules = {}
        for A, productions in self.rules.items():
            if A in reachable:
                filtered = []
                for prod in productions:
                    # если все символы prod достижимы, оставляем правило
                    if all(symbol in reachable for symbol in prod):
                        filtered.append(prod)
                if filtered:
                    new_rules[A] = filtered
        self.rules = new_rules

        return self

    def eliminate_cycles(self):
        """
        Устранение циклов в грамматике.

        Цикл - последовательность цепных правил, ведущих обратно к началу:
        A → B → C → A

        Шаги:
        1. Найти все циклы через цепные правила
        2. Объединить символы из каждого цикла
        3. Перестроить правила
        """
        print("Устранение циклов...")

        # Шаг 1: Построить граф цепных правил
        unit_graph = {}
        for left, rights in self.rules.items():
            unit_graph[left] = []
            for right in rights:
                # Цепное правило: A → B (один нетерминал)
                if len(right) == 1 and right in self.nonterminals:
                    unit_graph[left].append(right)

        # Найти циклы с помощью DFS
        visited = set()
        rec_stack = set()
        cycles = []

        def find_cycles_dfs(node, path):
            if node in rec_stack:
                # Найден цикл
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) > 1:  # Игнорируем самозациклы длиной 1
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            if node in unit_graph:
                for neighbor in unit_graph[node]:
                    find_cycles_dfs(neighbor, path[:])

            rec_stack.remove(node)

        # Запускаем поиск циклов для каждого нетерминала
        for nt in self.nonterminals:
            if nt not in visited:
                find_cycles_dfs(nt, [])

        if not cycles:
            print("Циклы не найдены")
            return self

        print(f"Найденные циклы: {cycles}")

        # Шаг 2: Объединить символы из циклов
        # Группируем пересекающиеся циклы
        cycle_groups = []
        for cycle in cycles:
            merged = False
            for group in cycle_groups:
                if any(symbol in group for symbol in cycle):
                    group.update(cycle)
                    merged = True
                    break
            if not merged:
                cycle_groups.append(set(cycle))

        # Шаг 3: Создать новые правила
        symbol_mapping = {}  # старый символ -> новый символ

        for group in cycle_groups:
            # Выбираем представителя группы (лексикографически первый)
            representative = min(group)
            for symbol in group:
                symbol_mapping[symbol] = representative

        print(f"Объединение символов: {symbol_mapping}")

        # Перестраиваем правила
        new_rules = {}
        new_nonterminals = set()

        for left, rights in self.rules.items():
            # Заменяем левую часть
            new_left = symbol_mapping.get(left, left)
            new_nonterminals.add(new_left)

            if new_left not in new_rules:
                new_rules[new_left] = []

            for right in rights:
                # Заменяем символы в правой части
                new_right = ''
                for symbol in right:
                    new_right += symbol_mapping.get(symbol, symbol)

                # Исключаем цепные правила внутри групп (A → A)
                if not (len(new_right) == 1 and new_right == new_left):
                    if new_right not in new_rules[new_left]:
                        new_rules[new_left].append(new_right)

        # Обновляем стартовый символ
        self.start = symbol_mapping.get(self.start, self.start)

        # Обновляем грамматику
        self.rules = new_rules
        self.nonterminals = new_nonterminals

        print("Циклы успешно устранены")
        return self

    def simplify_grammar(self):
        """
        Полное упрощение грамматики - применение всех алгоритмов в правильном порядке.

        Порядок важен:
        1. Устранение циклов
        2. Удаление epsilon-правил
        3. Удаление цепных правил
        4. Удаление бесполезных символов
        """
        print("Начинаем полное упрощение грамматики...")
        print("=" * 50)

        original_stats = self.get_statistics()

        self.remove_unit_productions()
        self.remove_unreachable_symbols()
        # Применяем алгоритмы в правильном порядке
        self.eliminate_cycles()

        # self.remove_epsilon_rules()
        # print("-" * 30)

        self.remove_useless_symbols()

        final_stats = self.get_statistics()

        print("\nРезультаты упрощения:")
        print(f"Правила: {original_stats['правила']} → {final_stats['правила']}")
        print(f"Нетерминалы: {original_stats['нетерминалы']} → {final_stats['нетерминалы']}")
        print(f"Терминалы: {original_stats['терминалы']} → {final_stats['терминалы']}")

        # Выводим дополнительные параметры для подробного сравнения
        print(
            f"Правила с многосимвольными левыми частями: {original_stats['многосимвольные_левые_части']} → {final_stats['многосимвольные_левые_части']}")
        print(f"Правила с эпсилон: {original_stats['правила_с_эпсилон']} → {final_stats['правила_с_эпсилон']}")

        print(f"Тип грамматики: {original_stats['тип_название']} → {final_stats['тип_название']}")

        print("=" * 50)
        print("Упрощение грамматики завершено!")

        return self

    def classify_grammar_symbols(self) -> None:
        """
        Классифицирует символы грамматики на терминалы и нетерминалы.
        Терминалы — маленькие буквы [a-z], нетерминалы — большие буквы [A-Z].
        Любой символ помимо букв считается ошибочным.
        Левые части правил могут содержать как большие, так и маленькие буквы,
        и будут добавлены в соответствующие множества.

        После вызова:
          self.terminals   — содержит все маленькие буквы из правил
          self.nonterminals — содержит все большие буквы из правил
        """
        terminals: set[str] = set()
        nonterminals: set[str] = set()

        # Обрабатываем левые части
        for left in self.rules:
            for symbol in left:
                if re.fullmatch(r'[A-Z]', symbol):
                    nonterminals.add(symbol)
                elif re.fullmatch(r'[a-z]', symbol):
                    terminals.add(symbol)
                elif symbol in self.nonterminals or symbol in self.terminals:
                    continue
                else:
                    raise ValueError(f"Недопустимый символ в левой части: '{symbol}' в нетерминале '{left}'")

        # Обрабатываем правые части
        for left, rights in self.rules.items():
            for right in rights:
                if right == 'ε':
                    continue
                for symbol in right:
                    if re.fullmatch(r'[A-Z]', symbol):
                        nonterminals.add(symbol)
                    elif re.fullmatch(r'[a-z]', symbol):
                        terminals.add(symbol)
                    elif symbol in self.nonterminals or symbol in self.terminals:
                        continue
                    else:
                        raise ValueError(f"Недопустимый символ в правиле: '{symbol}' в правой части '{right}'")

        # Обновляем множества в объекте
        self.terminals.update(terminals)
        self.nonterminals.update(nonterminals)

    def _auto_detect_symbols(self):
        """
        Автоматически определяет терминалы и нетерминалы из правил грамматики.

        УЛУЧШЕННАЯ ВЕРСИЯ с корректной обработкой составных токенов.
        """
        nonterminals = set()
        terminals = set()

        # Все левые части - нетерминалы
        for left in self.rules.keys():
            nonterminals.add(left)

        # Анализируем правые части
        for left, rights in self.rules.items():
            for right in rights:
                if right == 'ε' or right == '':
                    terminals.add('ε')
                    continue

                # Для правил в новом формате (с пробелами и специальными символами)
                if ' ' in right or any(char in right for char in '()[];,-{}'):
                    # Токенизируем правую часть по пробелам
                    tokens = self._tokenize_rule(right)

                    for token in tokens:
                        if token in self.rules:  # Если токен является левой частью правила
                            nonterminals.add(token)
                        else:
                            terminals.add(token)
                else:
                    # Классический формат - анализируем посимвольно
                    for symbol in right:
                        if symbol in self.rules:
                            nonterminals.add(symbol)
                        else:
                            terminals.add(symbol)

        return nonterminals, terminals

    def _tokenize_rule(self, rule_right):
        """
        УЛУЧШЕННАЯ токенизация правой части правила.
        Корректно обрабатывает составные токены как identifier, constant и др.
        """
        # Сначала разбиваем по пробелам
        tokens = rule_right.split()

        # Фильтруем и очищаем токены
        clean_tokens = []
        for token in tokens:
            if token:  # Убираем пустые токены
                clean_tokens.append(token)

        return clean_tokens  # Убираем пустые токены

    def update_symbols_detection(self):
        """
        Обновляет автоматическое определение терминалов и нетерминалов.
        Вызывается когда нужно пересчитать символы после изменения правил.
        """
        auto_nonterminals, auto_terminals = self._auto_detect_symbols()
        self.nonterminals.update(auto_nonterminals)
        self.terminals.update(auto_terminals)

        # Обновляем отсортированные правила
        self._sorted_rules = sorted(self.rules.items(), key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def compare_languages(solver1, solver2, n: int, method="length_limited"):
        """
        Генерирует n цепочек для grammar1 и grammar2,
        а затем выводит разность языков grammar1 grammar2

        :param solver1: Первый объект грамматики с методом generate_language(n)
        :param solver2: Второй объект грамматики с тем же методом
        :param n: Количество цепочек для генерации из каждой грамматики
        """
        # Генерируем цепочки из первой грамматики
        lang1 = set(solver1.generate_language(require_all_terminals=True, max_strings=n, method=method))
        print(f"Сгенерировано из первой грамматики ({len(lang1)} цепочек):")
        print(lang1)

        # Генерируем цепочки из второй грамматики
        lang2 = set(solver2.generate_language(require_all_terminals=True, max_strings=n, method=method))
        print(f"\nСгенерировано из второй грамматики ({len(lang2)} цепочек):")
        print(lang2)

        # Выполнение множества разности: что в first, но нет во second
        difference = lang1 - lang2

        print(f"\nЦепочки, которые есть в языке первой грамматики, но отсутствуют во второй ({len(difference)}):")
        print(difference)
        return difference

    def generate_language_with_spaces_support(self, max_length=100, max_strings=200, max_depth=100,
                                              require_all_terminals=True):
        """
        Модифицированный метод generate_language для поддержки правил с пробелами
        :param require_all_terminals: если True, в результат войдут только строки, содержащие все терминалы
        """
        from collections import deque

        def contains_only_terminals_spaces(string: str) -> bool:
            """Проверка терминальности с поддержкой пробелов"""
            if not string.strip():
                return True
            for nonterminal in self.nonterminals:
                if nonterminal in string:
                    return False
            return True

        def apply_rules_step_spaces(current_form: str):
            """Применение правил с поддержкой пробелов"""
            sorted_nonterminals = sorted(self.nonterminals, key=len, reverse=True)
            for nonterminal in sorted_nonterminals:
                if nonterminal in current_form:
                    results = []
                    for production in self.rules[nonterminal]:
                        replacement = production if production != 'ε' else ''
                        new_form = current_form.replace(nonterminal, replacement, 1)
                        results.append(new_form)
                    return results
            return []

        queue = deque([self.start])
        result = set()
        processed = set()
        iterations = 0
        max_iterations = min(max_strings * 10, 1000)

        while queue and len(result) < max_strings and iterations < max_iterations:
            iterations += 1
            current = queue.popleft()
            if current in processed:
                continue
            processed.add(current)

            # Если текущая форма терминальна
            if contains_only_terminals_spaces(current):
                if len(current) <= max_length:
                    if require_all_terminals:
                        # проверяем, что все терминалы присутствуют в строке
                        tokens = current.split()
                        if self.terminals.issubset(tokens):
                            result.add(current)
                    else:
                        result.add(current)
                continue

            # Иначе применяем правила
            next_forms = apply_rules_step_spaces(current)
            for form in next_forms:
                if form not in processed and len(form) <= max_length * 1.5:
                    queue.append(form)

        return sorted(result, key=lambda x: (len(x), x))

