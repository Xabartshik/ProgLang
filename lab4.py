from collections import defaultdict, deque

from lib.CFGParser import Grammar

# --- ДОБАВИТЬ в lab4.py рядом с импортами ---
from dataclasses import dataclass

@dataclass
class Node:
    sym: str
    children: list

def build_tree_from_left_derivation(g, rule_path):
    root = Node(g.start_symbol, [])
    form_syms = [g.start_symbol]
    form_nodes = [root]

    for nt, prod in rule_path:
        left_idx = next(i for i, s in enumerate(form_syms) if s in g.nonterminals)
        assert form_syms[left_idx] == nt, "Некорректный путь левого вывода"

        if len(prod) == 0:
            # визуализируем ε, но НЕ добавляем его в выравнивающий список узлов
            form_nodes[left_idx].children = [Node('ε', [])]
            # удаляем S из обоих выравнивающих списков
            form_syms  = form_syms[:left_idx]  + form_syms[left_idx+1:]
            form_nodes = form_nodes[:left_idx] + form_nodes[left_idx+1:]
        else:
            child_nodes = [Node(s, []) for s in prod]
            form_nodes[left_idx].children = child_nodes
            # поддерживаем строгую синхронизацию списков
            form_syms  = form_syms[:left_idx]  + prod        + form_syms[left_idx+1:]
            form_nodes = form_nodes[:left_idx] + child_nodes + form_nodes[left_idx+1:]

    return root


def print_tree_ascii(root):
    """Печатает дерево в стиле:
    S
    ├─ a
    └─ S
    """
    def _print(node, prefix, is_last):
        connector = "└─ " if is_last else "├─ "
        print(prefix + connector + node.sym)
        new_prefix = prefix + ("   " if is_last else "│  ")
        for i, ch in enumerate(node.children):
            _print(ch, new_prefix, i == len(node.children) - 1)

    # корень без коннектора
    print(root.sym)
    for i, ch in enumerate(root.children):
        _print(ch, "", i == len(root.children) - 1)


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
    print(" => ".join(forms))      # как было
    print("Дерево вывода:")        # новый блок
    tree = build_tree_from_left_derivation(g9, rule_path)
    print_tree_ascii(tree)

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

# G1: S → S1 | A0 ; A → A1 | 0
G1 = Grammar(
    nonterminals={'S','A'},
    terminals={'0','1'},
    productions={
        'S': [['S','1'], ['A','0']],
        'A': [['A','1'], ['0']]
    },
    start_symbol='S'
)

# G2: S → A1 | B0 | E1 ; A → S1 ; B → C1 | D1 ; C → 0 ; D → B1 ; E → E0 | 1
G2 = Grammar(
    nonterminals={'S','A','B','C','D','E'},
    terminals={'0','1'},
    productions={
        'S': [['A','1'], ['B','0'], ['E','1']],
        'A': [['S','1']],
        'B': [['C','1'], ['D','1']],
        'C': [['0']],
        'D': [['B','1']],
        'E': [['E','0'], ['1']]
    },
    start_symbol='S'
)

G1r = G1.to_right_linear_deterministic()
G2r = G2.to_right_linear_deterministic()



# Сравнение (генерируем строки до длины 20 и сравниваем множества)
print(G1.compare_generated_strings(G1r, max_length=5))
# Сравнение (аналогично)
print(G2.compare_generated_strings(G2r, max_length=5))

G1r.intersection(G2r)

G1.display_info()

print(G1.generate_strings(5))