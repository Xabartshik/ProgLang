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