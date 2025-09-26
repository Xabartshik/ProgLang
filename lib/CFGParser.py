from itertools import product
from typing import List, Dict, Set, Optional, Tuple

class CFGParser:
    def __init__(self, rules: Dict[str, List[str]], nonterminals: Optional[Set[str]] = None,
                 terminals: Optional[Set[str]] = None, start_symbol: str = 'S'):
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
        print(f"Инициализация CFGParser: nonterminals={self.nonterminals}, terminals={self.terminals}")

    def count_terminals(self, string: str) -> int:
        return sum(1 for c in string if c in self.terminals)

    def is_possible_prefix(self, current: str, target: str) -> bool:
        current_clean = ''.join(c for c in current if c in self.terminals)
        if len(current_clean) > len(target):
            return False
        return current_clean == target[:len(current_clean)]

    def generate_parse_trees(self, target: str, current_string: Optional[str] = None,
                             derivation: Optional[List[str]] = None) -> List[List[str]]:
        if current_string is None:
            current_string = self.start_symbol
        if derivation is None:
            derivation = [f"Start: {self.start_symbol}"]
        if current_string == target and all(c in self.terminals for c in current_string):
            return [derivation]
        if not self.is_possible_prefix(current_string, target):
            return []
        if self.count_terminals(current_string) > len(target):
            return []
        results = []
        i = 0
        while i < len(current_string):
            symbol = current_string[i]
            if symbol in self.nonterminals:
                for prod in self.rules.get(symbol, []):
                    new_string = current_string[:i] + prod + current_string[i + 1:]
                    step = f"{symbol} -> {prod if prod else 'ε'} at pos {i}"
                    results.extend(self.generate_parse_trees(target, new_string, derivation + [step]))
            i += 1
        return results

    def print_parse_trees(self, target: str):
        trees = self.generate_parse_trees(target)
        if not trees:
            print(f"Цепочка '{target}' не может быть выведена по данной грамматикой.")
            return
        print(f"\nВсе возможные деревья вывода для цепочки '{target}':")
        for i, tree in enumerate(trees, 1):
            print(f"\nДерево {i}:")
            for step in tree:
                print(step)

    def print_grammar_nice(self, rules: Dict[str, List[str]], header: str = "Грамматика"):
        print(f"\n{header}:")
        for nonterminal, productions in rules.items():
            print(f"{nonterminal} -> {' | '.join(prod if prod else 'ε' for prod in productions)}")

    def is_right_linear(self) -> bool:
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                print(f"Проверка правила: {nonterminal} -> {prod}")
                if prod == '':
                    print(f"Правило {nonterminal} -> ε допустимо")
                    continue
                if len(prod) > 2:
                    print(f"Правило {nonterminal} -> {prod} не праволинейное: длина > 2")
                    return False
                if len(prod) == 2 and (prod[0] not in self.terminals or prod[1] not in self.nonterminals):
                    print(f"Правило {nonterminal} -> {prod} не праволинейное: prod[0]={prod[0]} не терминал или prod[1]={prod[1]} не нетерминал")
                    return False
                if len(prod) == 1 and prod[0] not in self.terminals:
                    print(f"Правило {nonterminal} -> {prod} не праволинейное: prod[0]={prod[0]} не терминал")
                    return False
        return True

    def is_left_linear(self) -> bool:
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                print(f"Проверка правила: {nonterminal} -> {prod}")
                if prod == '':
                    print(f"Правило {nonterminal} -> ε допустимо")
                    continue
                if len(prod) > 2:
                    print(f"Правило {nonterminal} -> {prod} не леволинейное: длина > 2")
                    return False
                if len(prod) == 2 and (prod[0] not in self.nonterminals or prod[1] not in self.terminals):
                    print(f"Правило {nonterminal} -> {prod} не леволинейное: prod[0]={prod[0]} не нетерминал или prod[1]={prod[1]} не терминал")
                    return False
                if len(prod) == 1 and prod[0] not in self.terminals:
                    print(f"Правило {nonterminal} -> {prod} не леволинейное: prod[0]={prod[0]} не терминал")
                    return False
        return True

    def to_left_linear_grammar(self) -> Dict[str, List[str]]:
        if not self.is_right_linear():
            print("Грамматика не является праволинейной. Преобразование невозможно.")
            return {}
        nfa_states = self.nonterminals | {'F'}
        nfa_transitions = {state: {} for state in nfa_states}
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':
                    nfa_transitions[nonterminal].setdefault('', set()).add('F')
                elif len(prod) == 1:
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add('F')
                else:
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add(prod[1])
        dfa_states = set()
        dfa_transitions = {}
        state_queue = [frozenset([self.start_symbol])]
        dfa_states.add(frozenset([self.start_symbol]))
        final_states = set()
        while state_queue:
            current_state_set = state_queue.pop(0)
            dfa_transitions[current_state_set] = {}
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
            if 'F' in current_state_set:
                final_states.add(current_state_set)
        left_linear_rules = {}
        state_to_nonterminal = {state: f"Q{index}" for index, state in enumerate(dfa_states)}
        state_to_nonterminal[frozenset([self.start_symbol])] = self.start_symbol
        for state in dfa_states:
            nonterminal = state_to_nonterminal[state]
            left_linear_rules[nonterminal] = []
            if state in dfa_transitions:
                for terminal, next_state in dfa_transitions[state].items():
                    next_nonterminal = state_to_nonterminal[next_state]
                    left_linear_rules[nonterminal].append(f"{next_nonterminal}{terminal}")
            if state in final_states:
                left_linear_rules[nonterminal].append('')
        return left_linear_rules

    def to_right_linear_grammar(self) -> Dict[str, List[str]]:
        if not self.is_left_linear():
            print("Грамматика не является леволинейной. Преобразование невозможно.")
            return {}

        # Шаг 1: Построение НКА
        nfa_states = self.nonterminals | {'F'}
        nfa_transitions = {state: {} for state in nfa_states}
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':
                    nfa_transitions[nonterminal].setdefault('', set()).add('F')
                elif len(prod) == 1 and prod[0] in self.terminals:
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add('F')
                elif len(prod) == 2:
                    # Для леволинейной грамматики: A -> Bt
                    next_nonterminal, terminal = prod[0], prod[1]
                    nfa_transitions[next_nonterminal].setdefault(terminal, set()).add(nonterminal)

        print("НКА для леволинейной грамматики:", nfa_transitions)

        # Шаг 2: Построение ДКА
        dfa_states = set()
        dfa_transitions = {}
        initial_state = frozenset([self.start_symbol])
        state_queue = [initial_state]
        dfa_states.add(initial_state)
        final_states = set()

        while state_queue:
            current_state_set = state_queue.pop(0)
            dfa_transitions[current_state_set] = {}
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
            if 'F' in current_state_set:
                final_states.add(current_state_set)

        print("Состояния ДКА:", dfa_states)
        print("Финальные состояния ДКА:", final_states)
        print("Переходы ДКА:", dfa_transitions)

        # Шаг 3: Построение праволинейной грамматики
        right_linear_rules = {}
        state_to_nonterminal = {initial_state: self.start_symbol}
        available_nonterminals = [chr(65 + i) for i in range(26) if chr(65 + i) not in self.nonterminals]
        nonterminal_index = 0

        for state in dfa_states:
            if state != initial_state:
                if nonterminal_index < len(available_nonterminals):
                    state_to_nonterminal[state] = available_nonterminals[nonterminal_index]
                    nonterminal_index += 1
                else:
                    state_to_nonterminal[state] = f'Q{nonterminal_index}'
                    nonterminal_index += 1

        for state in dfa_states:
            nonterminal = state_to_nonterminal[state]
            right_linear_rules[nonterminal] = []
            if state in dfa_transitions:
                for terminal, next_state in dfa_transitions[state].items():
                    next_nonterminal = state_to_nonterminal[next_state]
                    right_linear_rules[nonterminal].append(f"{terminal}{next_nonterminal}")
            if state in final_states:
                right_linear_rules[nonterminal].append('')

        return right_linear_rules

    #НЕ РАБОТАЕТ НЕ РАБОТАЕТ

    def intersection_grammar(self, other: 'CFGParser') -> Dict[str, List[str]]:
        """
        Исправленная версия метода для построения пересечения двух грамматик
        """
        print(
            f"Проверка грамматик: self.is_right_linear={self.is_right_linear()}, self.is_left_linear={self.is_left_linear()}")
        print(
            f"Проверка грамматик: other.is_right_linear={other.is_right_linear()}, other.is_left_linear={other.is_left_linear()}")

        if not ((self.is_right_linear() or self.is_left_linear()) and (
                other.is_right_linear() or other.is_left_linear())):
            print("Обе грамматики должны быть либо праволинейными, либо леволинейными. Преобразование невозможно.")
            return {}

        # Построение НКА для первой грамматики
        nfa1_states = self.nonterminals | {'F1'}
        nfa1_transitions = {state: {} for state in nfa1_states}

        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':
                    nfa1_transitions[nonterminal].setdefault('', set()).add('F1')
                elif len(prod) == 1 and prod[0] in self.terminals:
                    nfa1_transitions[nonterminal].setdefault(prod[0], set()).add('F1')
                elif len(prod) == 2:
                    if self.is_right_linear():
                        nfa1_transitions[nonterminal].setdefault(prod[0], set()).add(prod[1])
                    else:  # Леволинейная
                        nfa1_transitions[prod[0]].setdefault(prod[1], set()).add(nonterminal)

        print("НКА для G1:", nfa1_transitions)

        # Построение НКА для второй грамматики
        nfa2_states = other.nonterminals | {'F2'}
        nfa2_transitions = {state: {} for state in nfa2_states}

        for nonterminal in other.rules:
            for prod in other.rules[nonterminal]:
                if prod == '':
                    nfa2_transitions[nonterminal].setdefault('', set()).add('F2')
                elif len(prod) == 1 and prod[0] in other.terminals:
                    nfa2_transitions[nonterminal].setdefault(prod[0], set()).add('F2')
                elif len(prod) == 2:
                    if other.is_right_linear():
                        nfa2_transitions[nonterminal].setdefault(prod[0], set()).add(prod[1])
                    else:  # Леволинейная
                        nfa2_transitions[prod[0]].setdefault(prod[1], set()).add(nonterminal)

        print("НКА для G2:", nfa2_transitions)

        # Пересечение НКА
        common_terminals = self.terminals & other.terminals
        if not common_terminals:
            print("Нет общих терминалов. Пересечение языков пусто.")
            return {}

        # Создание произведения состояний
        product_states = set()
        for s1 in nfa1_states:
            for s2 in nfa2_states:
                product_states.add((s1, s2))

        product_transitions = {(s1, s2): {} for s1, s2 in product_states}
        product_final_states = set((s1, s2) for s1, s2 in product_states if s1 == 'F1' and s2 == 'F2')

        for s1, s2 in product_states:
            for terminal in common_terminals:
                next_states1 = nfa1_transitions[s1].get(terminal, set())
                next_states2 = nfa2_transitions[s2].get(terminal, set())

                for ns1 in next_states1:
                    for ns2 in next_states2:
                        product_transitions[(s1, s2)].setdefault(terminal, set()).add((ns1, ns2))

        print("Переходы произведения НКА:", product_transitions)

        # Построение ДКА - ИСПРАВЛЕНИЕ: правильный формат начального состояния
        dfa_states = set()
        dfa_transitions = {}
        initial_state = frozenset([(self.start_symbol, other.start_symbol)])  # ИСПРАВЛЕНО: кортеж в множестве
        state_queue = [initial_state]
        dfa_states.add(initial_state)
        dfa_final_states = set()

        # ИСПРАВЛЕННОЕ Эпсилон-замыкание
        epsilon_closure = {}
        for s1, s2 in product_states:
            epsilon_closure[(s1, s2)] = {(s1, s2)}  # Инициализируем правильно

            # Добавляем epsilon-переходы, сохраняя формат кортежей
            if '' in nfa1_transitions.get(s1, {}):
                for ns1 in nfa1_transitions[s1]['']:
                    epsilon_closure[(s1, s2)].add((ns1, s2))

            if '' in nfa2_transitions.get(s2, {}):
                for ns2 in nfa2_transitions[s2]['']:
                    epsilon_closure[(s1, s2)].add((s1, ns2))

        # ИСПРАВЛЕННЫЙ основной цикл построения ДКА
        while state_queue:
            current_state_set = state_queue.pop(0)
            dfa_transitions[current_state_set] = {}

            for terminal in common_terminals:
                next_states = set()

                # ИСПРАВЛЕНИЕ: добавлена проверка типа элементов
                for state_pair in current_state_set:
                    if isinstance(state_pair, tuple) and len(state_pair) == 2:
                        s1, s2 = state_pair

                        # Получаем следующие состояния из произведения переходов
                        transitions = product_transitions.get((s1, s2), {}).get(terminal, set())
                        next_states.update(transitions)

                        # Добавляем эпсилон-замыкание для каждого нового состояния
                        for ns1, ns2 in transitions:
                            epsilon_states = epsilon_closure.get((ns1, ns2), set())
                            for eps_state in epsilon_states:
                                if isinstance(eps_state, tuple) and len(eps_state) == 2:
                                    next_states.add(eps_state)
                    else:
                        print(f"ПРЕДУПРЕЖДЕНИЕ: Некорректный элемент в состоянии: {state_pair}")

                if next_states:
                    next_state_set = frozenset(next_states)
                    if next_state_set not in dfa_states:
                        dfa_states.add(next_state_set)
                        state_queue.append(next_state_set)
                    dfa_transitions[current_state_set][terminal] = next_state_set

            # Проверяем, является ли состояние финальным
            is_final = False
            for state_pair in current_state_set:
                if isinstance(state_pair, tuple) and len(state_pair) == 2:
                    s1, s2 = state_pair
                    if (s1, s2) == ('F1', 'F2'):
                        is_final = True
                        break
                    # Проверяем эпсилон-замыкание
                    epsilon_states = epsilon_closure.get((s1, s2), set())
                    if ('F1', 'F2') in epsilon_states:
                        is_final = True
                        break

            if is_final:
                dfa_final_states.add(current_state_set)

        print("Состояния ДКА:", dfa_states)
        print("Финальные состояния ДКА:", dfa_final_states)
        print("Переходы ДКА:", dfa_transitions)

        # Формирование грамматики пересечения
        state_to_nonterminal = {initial_state: 'S'}
        available_nonterminals = [chr(65 + i) for i in range(26) if
                                  chr(65 + i) not in self.nonterminals | other.nonterminals]
        nonterminal_index = 0

        for state in dfa_states:
            if state != initial_state:
                if nonterminal_index < len(available_nonterminals):
                    state_to_nonterminal[state] = available_nonterminals[nonterminal_index]
                    nonterminal_index += 1
                else:
                    state_to_nonterminal[state] = f'Q{nonterminal_index - len(available_nonterminals)}'

        intersection_rules = {}
        for state in dfa_states:
            nonterminal = state_to_nonterminal[state]
            intersection_rules[nonterminal] = []

            if state in dfa_transitions:
                for terminal, next_state in dfa_transitions[state].items():
                    next_nonterminal = state_to_nonterminal[next_state]
                    intersection_rules[nonterminal].append(f"{terminal}{next_nonterminal}")

            if state in dfa_final_states:
                intersection_rules[nonterminal].append('')

        print("Созданные правила грамматики пересечения:")
        for nonterminal, productions in intersection_rules.items():
            print(f"{nonterminal} -> {' | '.join(productions)}")

        return intersection_rules

    def build_dfa(self) -> Tuple[Dict[frozenset, Dict[str, frozenset]], Set[frozenset], frozenset]:
        if not (self.is_right_linear() or self.is_left_linear()):
            print("Грамматика не является ни праволинейной, ни леволинейной. Построение ДКА невозможно.")
            return {}, set(), frozenset()

        nfa_states = self.nonterminals | {'F'}
        nfa_transitions = {state: {} for state in nfa_states}
        for nonterminal in self.rules:
            for prod in self.rules[nonterminal]:
                if prod == '':
                    nfa_transitions[nonterminal].setdefault('', set()).add('F')
                elif len(prod) == 1 and prod[0] in self.terminals:
                    nfa_transitions[nonterminal].setdefault(prod[0], set()).add('F')
                elif len(prod) == 2:
                    if self.is_right_linear():
                        nfa_transitions[nonterminal].setdefault(prod[0], set()).add(prod[1])
                    else:
                        nfa_transitions[prod[0]].setdefault(prod[1], set()).add(nonterminal)
        print("НКА для грамматики:", nfa_transitions)

        dfa_states = set()
        dfa_transitions = {}
        initial_state = frozenset([self.start_symbol])
        state_queue = [initial_state]
        dfa_states.add(initial_state)
        final_states = set()

        while state_queue:
            current_state_set = state_queue.pop(0)
            dfa_transitions[current_state_set] = {}
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
            if 'F' in current_state_set:
                final_states.add(current_state_set)
        print("Состояния ДКА:", dfa_states)
        print("Финальные состояния ДКА:", final_states)
        print("Переходы ДКА:", dfa_transitions)
        return dfa_transitions, final_states, initial_state

    def print_dfa(self, dfa_transitions: Dict[frozenset, Dict[str, frozenset]],
                  final_states: Set[frozenset], initial_state: frozenset):
        print("\nДетерминированный конечный автомат:")
        print("Состояния:", ", ".join(f"Q{index}" if s != initial_state else "S" for index, s in enumerate(dfa_transitions)))
        print("Начальное состояние: S")
        print("Финальные состояния:", ", ".join(f"Q{index}" for index, s in enumerate(dfa_transitions) if s in final_states))
        print("Переходы:")
        state_to_name = {state: f"Q{index}" for index, state in enumerate(dfa_transitions)}
        state_to_name[initial_state] = 'S'
        for state in dfa_transitions:
            for terminal, next_state in dfa_transitions[state].items():
                print(f"{state_to_name[state]} --{terminal}--> {state_to_name[next_state]}")

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