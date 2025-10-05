from collections import defaultdict, deque


class DFA:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.transitions = {}  # (state, sym) -> state
        self.start = None
        self.finals = set()

    def display(self):
        print(f"States: {self.states}")
        print(f"Alphabet: {self.alphabet}")
        print(f"Start: {self.start}")
        print(f"Finals: {self.finals}")
        print("Transitions:")
        for (state, sym), next_state in self.transitions.items():
            print(f"  delta({state}, {sym}) = {next_state}")


def intersect_dfa(dfa1, dfa2):
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("Алфавиты грамматик должны совпадать")

    alphabet = dfa1.alphabet
    product_states = set()
    product_trans = {}
    product_start = (dfa1.start, dfa2.start)
    queue = deque([product_start])
    product_states.add(product_start)

    while queue:
        cur1, cur2 = queue.popleft()
        for sym in alphabet:
            next1 = dfa1.transitions.get((cur1, sym))
            next2 = dfa2.transitions.get((cur2, sym))
            if next1 and next2:
                next_state = (next1, next2)
                product_trans[((cur1, cur2), sym)] = next_state
                if next_state not in product_states:
                    product_states.add(next_state)
                    queue.append(next_state)

    product_finals = {(s1, s2) for s1, s2 in product_states if s1 in dfa1.finals and s2 in dfa2.finals}

    # Переименовываем состояния (конкатенация для простоты, как SS, SA и т.д.)
    # ВАЖНО: читаемые имена пар состояний
    state_name = {(s1, s2): f'⟨{s1},{s2}⟩' for s1, s2 in product_states}

    dfa = DFA()
    dfa.states = set(state_name.values())
    dfa.alphabet = alphabet
    dfa.start = state_name[product_start]
    dfa.finals = {state_name[s] for s in product_finals}
    for ((cur1, cur2), sym), (n1, n2) in product_trans.items():
        dfa.transitions[(state_name[(cur1, cur2)], sym)] = state_name[(n1, n2)]

    return dfa


def grammar_from_dfa(dfa):
    nonterminals = dfa.states
    terminals = dfa.alphabet
    productions = defaultdict(list)
    for (state, sym), next_state in dfa.transitions.items():
        productions[state].append([sym, next_state])
    for final in dfa.finals:
        productions[final].append([])  # epsilon
    start = dfa.start
    return Grammar(nonterminals, terminals, dict(productions), start)


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
                # if state_name in self.nonterminals:
                #     state_name = 'Q_' + state_name
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

    def to_right_linear_deterministic(self):
        """
        Преобразует леволинейную регулярную грамматику в эквивалентную праволинейную регулярную грамматику
        с детерминированным разбором.

        Шаги:
        1. Построить эквивалентный NFA из леволинейной грамматики.
        2. Реверсировать NFA (для обратного языка).
        3. Детерминизировать реверсированный NFA в DFA.
        4. Преобразовать DFA в леволинейную грамматику для обратного языка.
        5. Зеркально отразить продукции, чтобы получить праволинейную грамматику для оригинального языка.
        """
        from collections import defaultdict, deque

        if not self.is_left_linear():
            raise ValueError("Грамматика должна быть леволинейной.")

        # Шаг 1: Построить NFA из леволинейной грамматики
        states = list(self.nonterminals) + ['F']
        transitions = defaultdict(set)  # (state, sym) -> set(states)
        epsilon = defaultdict(set)  # (state) -> set(states)
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
                    # A -> B x: A --x--> B
                    B, x = prod
                    if x not in self.terminals or B not in self.nonterminals:
                        raise ValueError("Некорректная продукция для регулярной грамматики")
                    transitions[(nt, x)].add(B)
                    alphabet.add(x)
                else:
                    raise ValueError("Продукция не соответствует леволинейной грамматике")

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

        # Шаг 4: Преобразовать DFA в леволинейную грамматику для rev(L)
        new_nonterminals = set(state_to_name.values())
        new_productions = defaultdict(list)
        new_start = state_to_name[start_set]

        for (current, sym), next_s in dfa_trans.items():
            nt = state_to_name[current]
            nt_next = state_to_name[next_s]
            # Леволинейная: A -> B x
            new_productions[nt].append([nt_next, sym])

        for acc_set in dfa_accept:
            nt = state_to_name[acc_set]
            if [] not in new_productions[nt]:
                new_productions[nt].append([])

        g_rev = Grammar(new_nonterminals, self.terminals, dict(new_productions), new_start)

        # Шаг 5: Зеркально отразить для получения праволинейной грамматики для L
        g_right = g_rev.mirror_productions()

        return g_right

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

    def to_nfa(self):
        if not self.is_right_linear():
            raise ValueError("Грамматика должна быть право-линейной для конвертации в NFA")

        states = list(self.nonterminals) + ['F']  # финальное состояние
        transitions = defaultdict(set)  # (state, sym or '') -> set of states
        alphabet = set()

        for nt, prods in self.productions.items():
            for prod in prods:
                if len(prod) == 0:
                    transitions[(nt, '')].add('F')
                elif len(prod) == 1:
                    x = prod[0]
                    if x not in self.terminals:
                        raise ValueError(f"Неверный терминал: {x}")
                    transitions[(nt, x)].add('F')
                    alphabet.add(x)
                elif len(prod) == 2:
                    x, B = prod
                    if x not in self.terminals or B not in self.nonterminals:
                        raise ValueError(f"Неверное производство: {prod}")
                    transitions[(nt, x)].add(B)
                    alphabet.add(x)
                else:
                    raise ValueError(f"Неверное производство: {prod}")

        start = self.start_symbol
        finals = {'F'}
        return states, alphabet, transitions, start, finals

    def determinize_nfa(self, states, alphabet, transitions, start, finals):
        def epsilon_closure(s_set):
            closure = set(s_set)
            queue = deque(s_set)
            while queue:
                s = queue.popleft()
                for t in transitions[(s, '')]:
                    if t not in closure:
                        closure.add(t)
                        queue.append(t)
            return frozenset(closure)

        dfa_states = set()
        dfa_trans = {}
        dfa_start = epsilon_closure({start})
        queue = deque([dfa_start])
        dfa_states.add(dfa_start)

        while queue:
            current = queue.popleft()
            for sym in alphabet:
                next_set = set()
                for state in current:
                    next_set.update(transitions[(state, sym)])
                next_closure = epsilon_closure(next_set)
                if next_closure:
                    dfa_trans[(current, sym)] = next_closure
                    if next_closure not in dfa_states:
                        dfa_states.add(next_closure)
                        queue.append(next_closure)

        dfa_finals = {s for s in dfa_states if finals & s}

        # Переименовываем состояния
        state_name = {s: ''.join(sorted(list(s))) if len(s) > 1 else (list(s)[0] if s else 'Empty') for s in dfa_states}
        for name in list(state_name.values()):
            if name in self.nonterminals:  # Избегаем конфликтов
                state_name = {k: 'Q' + v for k, v in state_name.items()}
                break

        dfa = DFA()
        dfa.states = set(state_name.values())
        dfa.alphabet = alphabet
        dfa.start = state_name[dfa_start]
        dfa.finals = {state_name[s] for s in dfa_finals}
        for (cur, sym), nxt in dfa_trans.items():
            dfa.transitions[(state_name[cur], sym)] = state_name[nxt]

        return dfa

    def to_dfa(self):
        states, alphabet, transitions, start, finals = self.to_nfa()
        return self.determinize_nfa(states, alphabet, transitions, start, finals)

    def intersection(self, other):
        """
        Вычисляет пересечение двух регулярных грамматик.
        Возвращает: (регулярная грамматика пересечения, DFA пересечения)
        """
        dfa1 = self.to_dfa()
        dfa2 = other.to_dfa()
        inter_dfa = intersect_dfa(dfa1, dfa2)
        inter_grammar = grammar_from_dfa(inter_dfa)
        return inter_grammar, inter_dfa


    def to_dfa_preserving_names(self):
        """
        Преобразует регулярную грамматику в DFA, максимально сохраняя имена нетерминалов
        """
        if self.is_left_linear():
            return self._left_to_dfa_preserving()
        elif self.is_right_linear():
            return self._right_to_dfa_preserving()
        else:
            raise ValueError("Грамматика должна быть регулярной")

    def _left_to_dfa_preserving(self):
        """
        Леволинейная грамматика → DFA с сохранением имен.

        Идея:
        - Для леволинейной грамматики правила имеют вид A → B x или A → x или A → ε.
          В NFA это интерпретируется как переход по символу x ИЗ B В A (стрелка «влево» в правиле).
        - Добавляется единое финальное состояние F для продукций с ε и одиночным терминалом.
        - Далее выполняется детерминизация (subset construction) с аккуратным именованием.

        Примеры отображения:
          A → B x   =>  B --x--> A
          A → x     =>  F --x--> A    (или A --x--> F, в зависимости от принятой конвенции; здесь выбираем F --x--> A,
                                       чтобы стартовать из F при детерминизации)
          A → ε     =>  ε-переход F --ε--> A
        """
        from collections import defaultdict, deque

        if not self.is_left_linear():
            raise ValueError("Ожидается леволинейная регулярная грамматика")

        F = "FINAL"
        transitions = defaultdict(set)  # (state, symbol) -> set(states)
        epsilon = defaultdict(set)  # state -> set(states)
        alphabet = set()

        # Построение NFA из леволинейной грамматики
        for A, prods in self.productions.items():
            for prod in prods:
                if len(prod) == 0:
                    # A → ε : F --ε--> A
                    epsilon[F].add(A)
                elif len(prod) == 1:
                    # A → x : F --x--> A
                    x = prod[0]
                    if x not in self.terminals:
                        raise ValueError(f"Ожидался терминал, получено: {prod}")
                    transitions[(F, x)].add(A)
                    alphabet.add(x)
                elif len(prod) == 2:
                    # A → B x : B --x--> A
                    B, x = prod
                    if B not in self.nonterminals or x not in self.terminals:
                        raise ValueError(f"Неверная продукция леволинейной грамматики: {prod}")
                    transitions[(B, x)].add(A)
                    alphabet.add(x)
                else:
                    raise ValueError(f"Продукция не соответствует леволинейной форме: {prod}")

        # ε-замыкание
        def epsilon_closure(states_set):
            closure = set(states_set)
            queue = deque(states_set)
            while queue:
                s = queue.popleft()
                for t in epsilon.get(s, []):
                    if t not in closure:
                        closure.add(t)
                        queue.append(t)
            return frozenset(closure)

        # Стартовое множество: ε-замыкание {F}
        start_set = epsilon_closure({F})

        # Аккуратное именование составных состояний
        def _canon_label(x):
            if isinstance(x, (set, frozenset)):
                inner = sorted(_canon_label(y) for y in x)
                return "{" + ",".join(inner) + "}"
            return str(x)

        state_to_name = {}
        queue = deque([start_set])
        seen = {start_set}

        while queue:
            current = queue.popleft()

            # Если это одно исходное состояние-нетерминал
            if len(current) == 1 and list(current)[0] in self.nonterminals:
                name = list(current)[0]
            elif current == frozenset({F}):
                name = F
            else:
                elems = sorted(_canon_label(s) for s in current if s != F)
                name = "{" + ",".join(elems) + "}"
                if F in current:
                    name += "_F"

            state_to_name[current] = name

            # Переходы по символам
            for sym in alphabet:
                next_states = set()
                for st in current:
                    next_states.update(transitions.get((st, sym), []))
                if next_states:
                    next_closure = epsilon_closure(next_states)
                    if next_closure not in seen:
                        seen.add(next_closure)
                        queue.append(next_closure)

        # Формируем DFA
        dfa = DFA()
        dfa.states = set(state_to_name.values())
        dfa.alphabet = alphabet
        dfa.start = state_to_name[start_set]

        # Финальными считаем те множества, где присутствует стартовый нетерминал грамматики,
        # поскольку в леволинейной интерпретации мы стартовали из F и «двигаемся к A».
        dfa.finals = {state_to_name[s] for s in seen if self.start_symbol in s}

        # Переходы DFA
        dfa_trans = {}
        for current in seen:
            cur_name = state_to_name[current]
            for sym in alphabet:
                next_states = set()
                for st in current:
                    next_states.update(transitions.get((st, sym), []))
                if next_states:
                    next_closure = epsilon_closure(next_states)
                    dfa_trans[(cur_name, sym)] = state_to_name[next_closure]

        dfa.transitions = dfa_trans
        return dfa

    def _left_to_dfa_preserving_no_final(self):
        """
        Леволинейная грамматика → DFA без явного состояния FINAL (виртуальный старт).
        """
        from collections import defaultdict, deque

        if not self.is_left_linear():
            raise ValueError("Ожидается леволинейная регулярная грамматика")  # [attached_file:8]

        # Транзиции между нетерминалами и стартовые «ходы»
        transitions = defaultdict(set)  # (B, x) -> {A} для правил A -> B x  [attached_file:8]
        start_moves = defaultdict(set)  # x -> {A} для правил A -> x        [attached_file:8]
        start_eps = set()  # {A} для правил A -> ε              [attached_file:8]
        alphabet = set()  # [attached_file:8]

        for A, prods in self.productions.items():
            for prod in prods:
                if len(prod) == 0:
                    start_eps.add(A)  # A -> ε  [attached_file:8]
                elif len(prod) == 1:
                    x = prod[0]
                    start_moves[x].add(A)  # A -> x  [attached_file:8]
                    alphabet.add(x)  # [attached_file:8]
                elif len(prod) == 2:
                    B, x = prod
                    transitions[(B, x)].add(A)  # A -> B x => B --x--> A  [attached_file:8]
                    alphabet.add(x)  # [attached_file:8]
                else:
                    raise ValueError(f"Не леволинейная продукция: {prod}")  # [attached_file:8]

        def epsilon_closure(states):
            # У леволинейной грамматики меж-нетерминальных ε-переходов нет; замыкание тождественно множеству [attached_file:8]
            return frozenset(states)  # [attached_file:8]

        start_set = epsilon_closure(start_eps)  # старт — ε-замыкание ε-правил  [attached_file:8]

        def canon_label(x):
            return "{" + ",".join(sorted(map(str, x))) + "}" if len(x) != 1 else next(iter(x))  # [attached_file:8]

        from collections import deque
        queue = deque([start_set])  # [attached_file:8]
        seen = {start_set}  # [attached_file:8]
        state_to_name = {start_set: canon_label(start_set) or "{}"}  # [attached_file:8]

        dfa_trans = {}  # (name, sym) -> name  [attached_file:8]

        while queue:
            cur = queue.popleft()  # [attached_file:8]
            cur_name = state_to_name[cur]  # [attached_file:8]
            for sym in alphabet:
                nxt = set()
                # обычные переходы из текущего множества
                for st in cur:
                    nxt.update(transitions.get((st, sym), ()))  # [attached_file:8]
                # плюс стартовые «ходы» только из стартового множества
                if cur is start_set:
                    nxt.update(start_moves.get(sym, ()))  # [attached_file:8]
                if not nxt:
                    continue  # [attached_file:8]
                nxt_fs = epsilon_closure(nxt)  # [attached_file:8]
                if nxt_fs not in state_to_name:
                    state_to_name[nxt_fs] = canon_label(nxt_fs)  # [attached_file:8]
                    seen.add(nxt_fs)  # [attached_file:8]
                    queue.append(nxt_fs)  # [attached_file:8]
                dfa_trans[(cur_name, sym)] = state_to_name[nxt_fs]  # [attached_file:8]

        # Принимающими делаем множества, содержащие исходный стартовый нетерминал
        finals = {state_to_name[s] for s in seen if self.start_symbol in s}  # [attached_file:8]

        dfa = DFA()  # [attached_file:8]
        dfa.states = set(state_to_name.values())  # [attached_file:8]
        dfa.alphabet = alphabet  # [attached_file:8]
        dfa.start = state_to_name[start_set]  # [attached_file:8]
        dfa.finals = finals  # [attached_file:8]
        dfa.transitions = dfa_trans  # [attached_file:8]
        return dfa  # [attached_file:8]

    def _right_to_dfa_preserving(self):
        """Праволинейная грамматика → DFA с сохранением имен"""
        from collections import defaultdict, deque

        # Добавляем только одно финальное состояние
        F = "FINAL"
        transitions = defaultdict(set)
        alphabet = set()

        # Строим NFA, используя исходные нетерминалы как состояния
        for nt, prods in self.productions.items():
            for prod in prods:
                if len(prod) == 0:  # A → ε
                    transitions[nt, ""].add(F)
                elif len(prod) == 1:  # A → a
                    x = prod[0]
                    transitions[nt, x].add(F)
                    alphabet.add(x)
                elif len(prod) == 2:  # A → aB
                    x, B = prod
                    transitions[nt, x].add(B)
                    alphabet.add(x)

        # Детерминизация с сохранением простых имен
        def epsilon_closure(states_set):
            closure = set(states_set)
            queue = deque(states_set)
            while queue:
                s = queue.popleft()
                for t in transitions.get((s, ""), []):
                    if t not in closure:
                        closure.add(t)
                        queue.append(t)
            return frozenset(closure)

        start_set = epsilon_closure({self.start_symbol})

        # Создаем читаемые имена для состояний DFA
        state_to_name = {}
        queue = deque([start_set])
        seen = {start_set}

        while queue:
            current = queue.popleft()

            # Если состояние содержит один исходный нетерминал, используем его имя
            if len(current) == 1 and list(current)[0] in self.nonterminals:
                state_name = list(current)[0]
            elif F in current and len(current) == 1:
                state_name = F
            else:
                # Для составных состояний используем множественную запись
                non_final = [s for s in current if s != F]
                if non_final:
                    state_name = "{" + ",".join(sorted(non_final)) + "}"
                    if F in current:
                        state_name += "_F"
                else:
                    state_name = F

            state_to_name[current] = state_name

            # Обрабатываем переходы
            for sym in alphabet:
                next_states = set()
                for state in current:
                    next_states.update(transitions.get((state, sym), []))

                if next_states:
                    next_closure = epsilon_closure(next_states)
                    if next_closure not in seen:
                        seen.add(next_closure)
                        queue.append(next_closure)

        # Создаем DFA
        dfa = DFA()
        dfa.states = set(state_to_name.values())
        dfa.alphabet = alphabet
        dfa.start = state_to_name[start_set]
        dfa.finals = {state_to_name[s] for s in seen if F in s}

        # Добавляем переходы
        dfa_trans = {}
        for current in seen:
            for sym in alphabet:
                next_states = set()
                for state in current:
                    next_states.update(transitions.get((state, sym), []))
                if next_states:
                    next_closure = epsilon_closure(next_states)
                    dfa_trans[(state_to_name[current], sym)] = state_to_name[next_closure]

        dfa.transitions = dfa_trans
        return dfa


    def intersection_preserving_names(self, other):
        """Пересечение с сохранением читаемых имен"""
        dfa1 = self.to_dfa_preserving_names()
        dfa2 = other.to_dfa_preserving_names()

        # Используем улучшенную функцию пересечения
        inter_dfa = intersect_dfa_readable(dfa1, dfa2)
        inter_grammar = grammar_from_dfa(inter_dfa)

        return inter_grammar, inter_dfa

# def intersect_dfa_readable(dfa1, dfa2):
#     """Пересечение DFA с читаемыми именами состояний"""
#     if dfa1.alphabet != dfa2.alphabet:
#         raise ValueError("Алфавиты должны совпадать")
#
#     alphabet = dfa1.alphabet
#     product_states = set()
#     product_trans = {}
#     product_start = (dfa1.start, dfa2.start)
#
#     queue = deque([product_start])
#     product_states.add(product_start)
#
#     while queue:
#         cur1, cur2 = queue.popleft()
#         for sym in alphabet:
#             next1 = dfa1.transitions.get((cur1, sym))
#             next2 = dfa2.transitions.get((cur2, sym))
#             if next1 and next2:
#                 next_state = (next1, next2)
#                 product_trans[((cur1, cur2), sym)] = next_state
#                 if next_state not in product_states:
#                     product_states.add(next_state)
#                     queue.append(next_state)
#
#     product_finals = {(s1, s2) for s1, s2 in product_states
#                       if s1 in dfa1.finals and s2 in dfa2.finals}
#
#     # Читаемые имена для пар состояний
#     state_name = {(s1, s2): f'⟨{s1},{s2}⟩' for s1, s2 in product_states}
#
#     dfa = DFA()
#     dfa.states = set(state_name.values())
#     dfa.alphabet = alphabet
#     dfa.start = state_name[product_start]
#     dfa.finals = {state_name[s] for s in product_finals}
#
#     for ((cur1, cur2), sym), (n1, n2) in product_trans.items():
#         dfa.transitions[(state_name[(cur1, cur2)], sym)] = state_name[(n1, n2)]
#
#     return dfa

def pretty_pair(s1, s2):
    a = [] if s1 == "FINAL" else [s1]                                   # [attached_file:8]
    b = [] if s2 == "FINAL" else [s2]                                   # [attached_file:8]
    if not a and not b:
        return "⟨START⟩"                                                # был ⟨FINAL,FINAL⟩  [attached_file:8]
    if not a:
        return f"⟨{b[0]}⟩"                                               # ⟨FINAL,X⟩ -> ⟨X⟩   [attached_file:8]
    if not b:
        return f"⟨{a[0]}⟩"                                               # ⟨X,FINAL⟩ -> ⟨X⟩   [attached_file:8]
    return f"⟨{a[0]},{b[0]}⟩"

def intersect_dfa_readable(dfa1, dfa2):
    """Пересечение DFA с читаемыми именами состояний + демонстрация всех пар."""
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("Алфавиты должны совпадать")

    alphabet = dfa1.alphabet
    product_states = set()
    product_trans = {}
    product_start = (dfa1.start, dfa2.start)

    queue = deque([product_start])
    product_states.add(product_start)

    print("Декартовы произведения переходов и проверка на пересечение:")
    while queue:
        cur1, cur2 = queue.popleft()
        for sym in alphabet:
            next1 = dfa1.transitions.get((cur1, sym))
            next2 = dfa2.transitions.get((cur2, sym))
            pair_desc = f"({cur1}, {cur2}) --{sym}--> ({next1}, {next2})"
            if next1 and next2:
                print(f"  {pair_desc}  -> принимается (обе ветки переходят)")
                next_state = (next1, next2)
                product_trans[((cur1, cur2), sym)] = next_state
                if next_state not in product_states:
                    product_states.add(next_state)
                    queue.append(next_state)
            else:
                reason = []
                if not next1:
                    reason.append(f"нет δ1({cur1},{sym})")
                if not next2:
                    reason.append(f"нет δ2({cur2},{sym})")
                print(f"  {pair_desc}  -> отбрасывается ({'; '.join(reason)})")

    product_finals = {
        (s1, s2)
        for s1, s2 in product_states
        if s1 in dfa1.finals and s2 in dfa2.finals
    }

    # Читаемые имена для пар состояний
    state_name = {(s1, s2): f'⟨{s1},{s2}⟩' for s1, s2 in product_states}
    # state_name = {(s1, s2): pretty_pair(s1, s2) for s1, s2 in product_states}

    dfa = DFA()
    dfa.states = set(state_name.values())
    dfa.alphabet = alphabet
    dfa.start = state_name[product_start]
    dfa.finals = {state_name[p] for p in product_finals}

    for ((cur1, cur2), sym), (n1, n2) in product_trans.items():
        dfa.transitions[(state_name[(cur1, cur2)], sym)] = state_name[(n1, n2)]

    return dfa
