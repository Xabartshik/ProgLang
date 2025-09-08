from collections import deque


class GrammarSolver:
    """
    Класс для решения задач по формальным грамматикам.
    Правила, алфавиты терминальных и нетерминальных символов задаются в конструкторе.
    Метод derive_string получает целевую цепочку и возвращает список шагов вывода.
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
            len(left) > 0 and all(right == '' or len(right) >= 1 for right in rights)
            for left, rights in self.rules.items()
        )
        if is_context_sensitive:
            return 1

        return 0

