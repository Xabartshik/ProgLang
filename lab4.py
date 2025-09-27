from collections import defaultdict, deque


class Grammar:
    def __init__(self, nonterminals, terminals, productions, start_symbol):
        """
        Инициализация формальной грамматики.

        :param nonterminals: Множество нетерминальных символов (строки).
        :param terminals: Множество терминальных символов (строки, могут быть многосимвольными).
        :param productions: Словарь: ключ — нетерминал (строка), значение — список списков строк
                            (каждый внутренний список — это правая часть правила, элементы — терминалы или нетерминалы).
        :param start_symbol: Начальный нетерминал (строка).
        """
        self.nonterminals = set(nonterminals)
        self.terminals = set(terminals)
        self.productions = productions  # dict: нетерминал -> список [символы правой части]
        self.start_symbol = start_symbol

        # Проверка корректности грамматики
        self._validate()

    def _validate(self):
        """Проверяет компоненты грамматики на корректность."""
        all_symbols = self.nonterminals.union(self.terminals)
        if self.start_symbol not in self.nonterminals:
            raise ValueError("Стартовый символ должен быть нетерминалом.")
        for nt, prods in self.productions.items():
            if nt not in self.nonterminals:
                raise ValueError(f"Ключ {nt} в продукциях не является нетерминалом.")
            for prod in prods:
                for sym in prod:
                    if sym not in all_symbols:
                        raise ValueError(f"Символ {sym} в правиле {nt} -> {prod} не определён.")

    def is_regular(self):
        """
        Проверяет, является ли грамматика регулярной (леволинейной или праволинейной).
        """
        is_left_linear = True
        is_right_linear = True
        for nt, prods in self.productions.items():
            for prod in prods:
                # Праволинейная: A -> aB или A -> a (максимум один нетерминал в конце)
                if len(prod) > 2 or (len(prod) == 2 and prod[0] in self.nonterminals) or \
                        (len(prod) == 1 and prod[0] in self.nonterminals):
                    is_right_linear = False
                # Леволинейная: A -> Ba или A -> a (максимум один нетерминал в начале)
                if len(prod) > 2 or (len(prod) == 2 and prod[1] in self.nonterminals) or \
                        (len(prod) == 1 and prod[0] in self.nonterminals):
                    is_left_linear = False
        return is_left_linear or is_right_linear

    def is_left_linear(self):
        """Проверяет, является ли грамматика леволинейной."""
        for nt, prods in self.productions.items():
            for prod in prods:
                if len(prod) > 2 or (len(prod) == 2 and prod[1] in self.nonterminals) or \
                        (len(prod) == 1 and prod[0] in self.nonterminals):
                    return False
        return True

    def is_right_linear(self):
        """Проверяет, является ли грамматика праволинейной."""
        for nt, prods in self.productions.items():
            for prod in prods:
                if len(prod) > 2 or (len(prod) == 2 and prod[0] in self.nonterminals) or \
                        (len(prod) == 1 and prod[0] in self.nonterminals):
                    return False
        return True

    def mirror_productions(self):
        """
        Зеркально разворачивает правые части правил для преобразования между леволинейной и праволинейной грамматикой.
        Разворачивает каждую правую часть наоборот. Начальный символ не меняется.
        Важно: предполагается, что грамматика регулярная.
        """
        if not self.is_regular():
            raise ValueError("Грамматика должна быть регулярной для зеркального отражения.")

        new_productions = {}
        for nt, prods in self.productions.items():
            new_productions[nt] = [prod[::-1] for prod in prods]

        return Grammar(self.nonterminals, self.terminals, new_productions, self.start_symbol)

    def display_info(self):
        """Выводит информацию о грамматике: нетерминалы, терминалы и продукции."""
        print("Нетерминалы:", self.nonterminals)
        print("Терминалы:", self.terminals)
        print("Стартовый символ:", self.start_symbol)
        print("Продукции:")
        for nt, prods in sorted(self.productions.items()):
            for prod in prods:
                print(f"  {nt} -> {'ε' if not prod else ' '.join(prod)}")

    from collections import deque

    def generate_strings(self, max_length=10):
        """
        Генерирует множество строк языка грамматики до максимальной длины max_length.
        Возвращает множество строк (конкатенация терминалов).
        Поддерживает RHS как списки лексем, так и строки (включая '' как ε).
        """

        # Подготовим список всех допустимых лексем для токенизации строковых RHS
        all_symbols = list(self.nonterminals | self.terminals)
        # Для жадной токенизации — сортируем по убыванию длины
        all_symbols.sort(key=len, reverse=True)

        def tokenize_rhs(rhs):
            """
            Преобразует правую часть правила к списку лексем.
            Допускаются форматы:
              - list[str]    -> возвращается как есть
              - tuple[str]   -> list(rhs)
              - str          -> жадная токенизация по all_symbols; '' => []
            """
            if isinstance(rhs, list):
                return rhs
            if isinstance(rhs, tuple):
                return list(rhs)
            if isinstance(rhs, str):
                if rhs == '':
                    return []  # ε
                i = 0
                out = []
                while i < len(rhs):
                    matched = False
                    for sym in all_symbols:
                        if rhs.startswith(sym, i):
                            out.append(sym)
                            i += len(sym)
                            matched = True
                            break
                    if not matched:
                        # Если токенизация невозможна, считаем каждый символ отдельной лексемой
                        # (как fallback для односимвольных алфавитов)
                        out.append(rhs[i])
                        i += 1
                return out
            # Неподдерживаемый тип — приводим через str и токенизируем
            return tokenize_rhs(str(rhs))

        result = set()
        # Цепочка хранится как список лексем
        queue = deque([([self.start_symbol], 0)])
        visited = set()

        def terminals_len(chain):
            # Длина итоговой строки — число терминальных лексем
            return sum(1 for s in chain if s in self.terminals)

        max_depth = 2 * max_length  # простая отсечка

        while queue:
            current_chain, depth = queue.popleft()
            state = (tuple(current_chain), depth)
            if state in visited:
                continue
            visited.add(state)

            if all(sym in self.terminals for sym in current_chain):
                if terminals_len(current_chain) <= max_length:
                    result.add(''.join(current_chain))
                continue

            # Находим первый нетерминал (левая деривация)
            nt_pos = None
            nt_sym = None
            for i, sym in enumerate(current_chain):
                if sym in self.nonterminals:
                    nt_pos, nt_sym = i, sym
                    break
            if nt_pos is None:
                continue

            for raw_prod in self.productions.get(nt_sym, []):
                prod = tokenize_rhs(raw_prod)
                new_chain = current_chain[:nt_pos] + prod + current_chain[nt_pos + 1:]

                # Отсечка по глубине
                if depth + 1 > max_depth:
                    continue

                # Грубая отсечка по длине терминалов в частичных формах
                # (не строгая, но помогает не раздувать очередь)
                if terminals_len(new_chain) <= max_length:
                    queue.append((new_chain, depth + 1))
                else:
                    # Если уже превысили целевую длину терминалов и в цепочке нет нетерминалов, смысла продолжать нет
                    if all(s in self.terminals for s in new_chain):
                        continue

        return {s for s in result if len(s) <= max_length}

    def compare_generated_strings(self, other, max_length=10):
        """
        Сравнивает множества сгенерированных строк двух грамматик до указанной максимальной длины.

        :param other: Другая грамматика для сравнения.
        :param max_length: Максимальная длина сохраняемых строк.
        :return: True, если множества строк совпадают, иначе False.
        """
        strings_self = self.generate_strings(max_length)
        strings_other = other.generate_strings(max_length)
        return strings_self == strings_other



    def to_left_linear_deterministic(self):
        """
        Преобразует праволинейную регулярную грамматику в эквивалентную леволинейную регулярную грамматику
        с детерминированным разбором.
        Шаги:
        1. Построить эквивалентный NFA из праволинейной грамматики.
        2. Реверсировать NFA (для обратного языка).
        3. Детерминизировать реверсированный NFA в DFA.
        5. Преобразовать DFA в праволинейную грамматику для обратного языка.
        6. Зеркально отразить продукций, чтобы получить леволинейную грамматику для оригинального языка.
        """
        if not self.is_right_linear():
            raise ValueError("Грамматика должна быть праволинейной.")

        # Шаг 1: Построить NFA
        # Состояния: нетерминалы + финальное 'F'
        states = list(self.nonterminals) + ['F']
        # Переходы: (состояние, символ) -> set(состояний)
        transitions = defaultdict(set)
        # Eps-переходы: состояние -> set(состояний)
        epsilon = defaultdict(set)
        # Алфавит
        alphabet = set()

        for nt, prods in self.productions.items():
            for prod in prods:
                if len(prod) == 0:
                    # A -> ε: A --ε--> F
                    epsilon[nt].add('F')
                elif len(prod) == 1:
                    # A -> x: A --x--> F, x терминал
                    x = prod[0]
                    if x not in self.terminals:
                        raise ValueError("Некорректная продукция для регулярной грамматики")
                    transitions[(nt, x)].add('F')
                    alphabet.add(x)
                elif len(prod) == 2:
                    # A -> x B: A --x--> B
                    x, B = prod
                    if x not in self.terminals or B not in self.nonterminals:
                        raise ValueError("Некорректная продукция для регулярной грамматики")
                    transitions[(nt, x)].add(B)
                    alphabet.add(x)
                else:
                    raise ValueError("Продукция не соответствует праволинейной грамматике")

        # Шаг 2: Реверсировать NFA
        rev_trans = defaultdict(set)
        rev_epsilon = defaultdict(set)

        for (p, sym), qs in transitions.items():
            for q in qs:
                rev_trans[(q, sym)].add(p)

        for p, qs in epsilon.items():
            for q in qs:
                rev_epsilon[q].add(p)

        # Новый старт: 'F', новый accept: start_symbol

        # Шаг 3: Детерминизировать реверсированный NFA
        def epsilon_closure(states_set, eps_dict):
            """Вычисляет ε-замыкание для множества состояний."""
            closure = set(states_set)
            queue = deque(states_set)
            while queue:
                s = queue.popleft()
                for t in eps_dict[s]:
                    if t not in closure:
                        closure.add(t)
                        queue.append(t)
            return frozenset(closure)

        # Начальное множество состояний: ε-замыкание {'F'}
        start_set = epsilon_closure({'F'}, rev_epsilon)

        # DFA состояния: список frozenset
        dfa_states = []
        state_to_name = {}
        next_state_id = 0

        queue = deque([start_set])
        seen = {start_set}
        dfa_trans = {}  # (frozenset, sym) -> frozenset

        while queue:
            current = queue.popleft()
            # Присваиваем имя состоянию, если еще не
            if current not in state_to_name:
                state_name = ''.join(sorted(current)) if len(current) > 1 else (
                    list(current)[0] if current else 'Empty')
                # Чтобы избежать конфликтов, добавляем префикс если нужно
                if state_name in self.nonterminals:
                    state_name = 'Q_' + state_name
                state_to_name[current] = state_name
                dfa_states.append(current)

            for sym in alphabet:
                next_states = set()
                for state in current:
                    next_states.update(rev_trans[(state, sym)])
                if next_states:
                    next_closure = epsilon_closure(next_states, rev_epsilon)
                    dfa_trans[(current, sym)] = next_closure
                    if next_closure not in seen:
                        seen.add(next_closure)
                        queue.append(next_closure)

        # DFA принимающие состояния: те, что содержат оригинальный start_symbol
        dfa_accept = [s for s in seen if self.start_symbol in s]

        # Шаг 5: Преобразовать DFA в праволинейную грамматику для rev(L)
        new_nonterminals = set(state_to_name.values())
        new_productions = defaultdict(list)
        new_start = state_to_name[start_set]

        for (current, sym), next_s in dfa_trans.items():
            nt = state_to_name[current]
            prod = [sym, state_to_name[next_s]]
            new_productions[nt].append(prod)

        for acc_set in dfa_accept:
            nt = state_to_name[acc_set]
            if [] not in new_productions[nt]:
                new_productions[nt].append([])

        g_rev = Grammar(new_nonterminals, self.terminals, dict(new_productions), new_start)

        # Шаг 6: Зеркально отразить для получения леволинейной грамматики для L
        g_left = g_rev.mirror_productions()

        return g_left

    def all_left_derivations(self, tokens: list[str]):
        """
        Находит все возможные левые выводы для данной цепочки токенов с помощью симуляции недетерминированного
        магазинного автомата (PDA). Возвращает список последовательностей примененных правил, где каждое правило
        представлено как кортеж (нетерминал, продукция как список строк).
        """
        from collections import deque

        results = []
        queue = deque()
        # Начальное состояние: позиция 0, стек с начальным символом (bottom to top)
        queue.append((0, [self.start_symbol], []))  # (pos, stack, rule_path)

        while queue:
            pos, stack, rule_path = queue.popleft()

            # Если стек пуст и весь ввод обработан — принимаем
            if len(stack) == 0:
                if pos == len(tokens):
                    results.append(rule_path[:])
                continue

            # Верх стека
            top = stack[-1]

            if top in self.terminals:
                # Если терминал — пытаемся сопоставить с текущим токеном
                if pos < len(tokens) and top == tokens[pos]:
                    new_stack = stack[:-1]
                    queue.append((pos + 1, new_stack, rule_path))
            elif top in self.nonterminals:
                # Если нетерминал — заменяем на каждую продукцию (push в обратном порядке)
                for prod in self.productions.get(top, []):
                    new_stack = stack[:-1] + prod[::-1]
                    new_rule_path = rule_path + [(top, prod)]
                    queue.append((pos, new_stack, new_rule_path))

        return results



def reconstruct_derivation(g, rule_path, tokens):
    """
    Восстанавливает последовательность сентенциальных форм из пути правил.
    Возвращает список строк (конкатенированных форм).
    """
    forms = []
    form = [g.start_symbol]
    forms.append(''.join(form))  # Начальная форма

    for nt, prod in rule_path:
        # Находим левый нетерминал
        left_idx = next((i for i, sym in enumerate(form) if sym in g.nonterminals), None)
        if left_idx is None or form[left_idx] != nt:
            raise ValueError("Некорректный путь вывода")
        form = form[:left_idx] + prod + form[left_idx + 1:]
        forms.append(''.join(form))

    # Проверяем, что финальная форма соответствует токенам
    if form != tokens:
        raise ValueError("Вывод не генерирует заданную цепочку")
    return forms





# Пример для задачи 9
nonterminals_9 = {'S'}
terminals_9 = {'a', 'b'}
productions_9 = {
    'S': [['a', 'S', 'b', 'S'], ['b', 'S', 'a', 'S'], []]
}
start_symbol_9 = 'S'

g9 = Grammar(nonterminals_9, terminals_9, productions_9, start_symbol_9)
tokens_9 = ['a', 'b', 'a', 'b']  # Цепочка abab

print("\nРешение задачи 9: Все левые выводы для цепочки abab")
derivations = g9.all_left_derivations(tokens_9)

for idx, rule_path in enumerate(derivations, 1):
    print(f"\nВывод {idx}:")
    forms = reconstruct_derivation(g9, rule_path, tokens_9)
    print(" => ".join(forms))

nonterminals_a = {'S', 'B', 'C'}
terminals_a = {'0', '1', '⊥'}
productions_a = {
    'S': [['0', 'S'], ['0', 'B']],
    'B': [['1', 'B'], ['1', 'C']],
    'C': [['1', 'C'], ['⊥']]
}
start_symbol_a = 'S'

# g_a = Grammar(nonterminals_a, terminals_a, productions_a, start_symbol_a)
# g_left_a = g_a.to_left_linear_deterministic()
#
# print("Новая леволинейная грамматика для варианта a:")
# print("Нетерминалы:", g_left_a.nonterminals)
# print("Терминалы:", g_left_a.terminals)
# print("Старт:", g_left_a.start_symbol)
# print("Продукции:")
# for nt, prods in sorted(g_left_a.productions.items()):
#     for prod in prods:
#         print(f"{nt} -> {' '.join(prod) if prod else 'ε'}")
#
# # Пример использования для грамматики b из задачи 11
# nonterminals_b = {'S', 'A', 'B'}
# terminals_b = {'a', 'b', '⊥'}
# productions_b = {
#     'S': [['a', 'A'], ['a', 'B'], ['b', 'A']],
#     'A': [['b', 'S']],
#     'B': [['a', 'S'], ['b', 'B'], ['⊥']]
# }
# start_symbol_b = 'S'
#
# g_b = Grammar(nonterminals_b, terminals_b, productions_b, start_symbol_b)
# g_left_b = g_b.to_left_linear_deterministic()
#
# print("\nНовая леволинейная грамматика для варианта b:")
# print("Нетерминалы:", g_left_b.nonterminals)
# print("Терминалы:", g_left_b.terminals)
# print("Старт:", g_left_b.start_symbol)
# print("Продукции:")
# for nt, prods in sorted(g_left_b.productions.items()):
#     for prod in prods:
#         print(f"{nt} -> {' '.join(prod) if prod else 'ε'}")

# Праволинейная грамматика для варианта a
g_right_a = Grammar(
    nonterminals={'S', 'B', 'C'},
    terminals={'0', '1'},
    productions={
        'S': ['0S', '0B'],
        'B': ['1B', '1C'],
        'C': ['1C', '']
    },
    start_symbol='S'
)




# Праволинейная грамматика для варианта б
g_right_b = Grammar(
    nonterminals={'S', 'A', 'B'},
    terminals={'a', 'b'},
    productions={
        'S': ['aA', 'aB', 'bA'],
        'A': ['bS'],
        'B': ['aS', 'bB', '']
    },
    start_symbol='S'
)

g_left_a = g_right_a.to_left_linear_deterministic()
g_left_b = g_right_b.to_left_linear_deterministic()

# Сравнение (генерируем строки до длины 20 и сравниваем множества)
print(g_right_a.compare_generated_strings(g_left_a, max_length=5))
# Сравнение (аналогично)
print(g_right_b.compare_generated_strings(g_left_b, max_length=5))
