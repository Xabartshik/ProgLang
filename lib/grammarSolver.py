from collections import deque
import random
from typing import List, Set, Tuple, Optional


class GrammarSolver:
    """
    Класс для решения задач по формальным грамматикам.
    Правила, алфавиты терминальных и нетерминальных символов задаются в конструкторе.
    Метод derive_string получает целевую цепочку и возвращает список шагов вывода.
    Добавлен метод generate_language для генерации строк языка.
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

    def derive_string(self, target: str) -> list[str]:
        """
        Построение вывода для заданной цепочки с использованием обхода в ширину.
        :param target: Целевая цепочка (строка).
        :return: Список шагов вывода или ["Вывод не найден"].
        """
        from collections import deque

        queue = deque([(self.start, [self.start])])
        visited = set([self.start])
        target_symbols = set(target)
        non_terminal_symbols = self.terminals - target_symbols
        if non_terminal_symbols:
            return [
                f"Ошибка ЯЗЫК НЕ ТОТ: в целевой цепочке '{target}' отсутствуют символы из алфавита терминалов: {sorted(non_terminal_symbols)}"]

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            # Применяем правила только к leftmost вхождению для каждого правила (left - символ для замены, right - на что менять)
            for left, rights in self.rules.items():
                # Перебор всех правил
                for right in rights:
                    # Поиск возможности применить правило
                    pos = current.find(left)
                    if pos != -1:
                        # Вставка с заменой подстроки (символы до позиции, вставка правила, символы после)
                        new_current = current[:pos] + right + current[pos + len(left):]
                        if new_current not in visited:
                            # Добавляем в посещенные
                            visited.add(new_current)
                            # Добавляем в список перебора
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
                    if right == '':
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
                    if right == '':
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
                (right == '' and left == self.start) or
                (right != '' and len(right) >= len(left))
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

        :param method: Метод генерации ('exhaustive' для исчерпывающего, 'random' для случайного).
        :param max_length: Максимальная длина генерируемых строк.
        :param max_strings: Максимальное количество строк для генерации.
        :param max_depth: Максимальная глубина рекурсии для случайной генерации.
        :return: Отсортированный список строк языка.
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

        :param max_length: Максимальная длина строки.
        :param max_strings: Максимальное количество строк.
        :return: Отсортированный список строк языка.
        """
        queue = deque([(self.start,)])  # Кортеж символов
        result = set()
        processed = set()

        while queue and len(result) < max_strings:
            current_form = queue.popleft()

            # Преобразуем в строку для проверки на дубликаты
            form_str = ''.join(current_form)
            if form_str in processed:
                continue
            processed.add(form_str)

            # Если все символы терминальные - добавляем в результат
            if all(symbol in self.terminals for symbol in current_form):
                terminal_string = ''.join(current_form)
                if len(terminal_string) <= max_length:
                    result.add(terminal_string)
                continue

            # Находим первый нетерминал для замены
            for i, symbol in enumerate(current_form):
                if symbol in self.nonterminals:
                    if symbol in self.rules:
                        # Применяем все правила для этого нетерминала
                        for rule in self.rules[symbol]:
                            # Преобразуем правило в список символов
                            rule_symbols = list(rule) if rule else []
                            new_form = current_form[:i] + tuple(rule_symbols) + current_form[i + 1:]

                            # Проверяем потенциальную минимальную длину
                            min_length = sum(1 for s in new_form if s in self.terminals)
                            if min_length <= max_length and len(new_form) <= max_length * 2:
                                queue.append(new_form)
                    break

        return sorted(result, key=lambda x: (len(x), x))

    def _generate_language_random(self, count: int, max_depth: int, max_length: int) -> List[str]:
        """
        Генерирует случайные строки языка.

        :param count: Количество строк для генерации.
        :param max_depth: Максимальная глубина рекурсии.
        :param max_length: Максимальная длина строки.
        :return: Отсортированный список строк языка.
        """
        result = set()
        attempts = 0
        max_attempts = count * 50

        while len(result) < count and attempts < max_attempts:
            try:
                string = self._generate_random_string(self.start, max_depth)
                if string and len(string) <= max_length and string not in result:
                    result.add(string)
            except (RecursionError, Exception):
                # Игнорируем ошибки при генерации
                pass
            attempts += 1

        return sorted(result, key=lambda x: (len(x), x))

    def _generate_random_string(self, symbol: str, max_depth: int) -> str:
        """
        Рекурсивная генерация случайной строки из грамматики.

        :param symbol: Текущий символ для развертывания.
        :param max_depth: Оставшаяся глубина рекурсии.
        :return: Сгенерированная строка.
        """
        if max_depth <= 0:
            return ""

        if symbol in self.terminals:
            return symbol

        if symbol not in self.rules:
            return symbol

        # Выбираем правило для применения
        rules = self.rules[symbol]

        # При малой глубине предпочитаем терминальные правила
        if max_depth <= 3:
            terminal_rules = [rule for rule in rules
                              if all(s in self.terminals for s in rule)]
            if terminal_rules:
                rules = terminal_rules

        chosen_rule = random.choice(rules)

        # Применяем выбранное правило
        result = ""
        for s in chosen_rule:
            result += self._generate_random_string(s, max_depth - 1)

        return result

    def format_language_output(self, grammar_name: str, language_description: str,
                               strings: List[str]) -> str:
        """
        Форматирует вывод языка в стиле L(G) = {...}.

        :param grammar_name: Название грамматики (например, "G1").
        :param language_description: Описание языка (например, "{0^n 1^n | n>0}").
        :param strings: Список строк языка.
        :return: Отформатированная строка вывода.
        """
        output = f"L({grammar_name}) = {language_description}\n"
        for string in strings:
            output += f"  {string}\n"
        return output

    def print_grammar(self):
        """
        Выводит грамматику в читаемом виде.
        """
        print(f"Стартовый символ: {self.start}")
        print(f"Нетерминалы: {self.nonterminals}")
        print(f"Терминалы: {self.terminals}")
        print("Правила грамматики:")
        for left, rights in self.rules.items():
            rights_str = " | ".join(rights)
            print(f"  {left} -> {rights_str}")
        print()

    def get_grammar_type_name(self) -> str:
        """
        Возвращает название типа грамматики по классификации Хомского.

        :return: Название типа грамматики.
        """
        type_num = self.get_grammar_type()
        type_names = {
            0: "Неограниченная (тип 0)",
            1: "Контекстно-зависимая (тип 1)",
            2: "Контекстно-свободная (тип 2)",
            3: "Регулярная (тип 3)"
        }
        return type_names.get(type_num, "Неизвестный тип")

    def generate_description(self) -> str:
        """
        Попытка автоматически сформировать описание языка в нотации {… | …}
        на основании структуры грамматики и её типа.
        Для регулярных грамматик выдаёт вид {a^n b^m … | n>0, m>0, …};
        для остальных — общее описание.
        """
        t = self.get_grammar_type()
        # Регулярная грамматика (тип 3)
        if t == 3:
            # собираем все терминалы, задействованные в правилах
            used = set()
            for rights in self.rules.values():
                for r in rights:
                    used.update(ch for ch in r if ch in self.terminals)
            # для каждого терминала вводим счётчик >0
            parts = []
            for sym in sorted(used):
                parts.append(f"{sym}^n_{sym}")
            cond = ", ".join(f"n_{sym}>0" for sym in sorted(used))
            base = " ".join(parts)
            return f"{{{base} | {cond}}}"
        # Контекстно-свободная (тип 2)
        if t == 2:
            return "{строки, порождаемые КС-грамматикой}"
        # Контекстно-зависимая (тип 1)
        if t == 1:
            return "{строки, порождаемые контекстно-зависимой грамматикой}"
        # Неограниченная (тип 0) или неизвестный
        return "{любой язык, порождаемый этой грамматикой}"

