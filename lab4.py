from collections import defaultdict, deque

from lib.CFGParser import Grammar

from dataclasses import dataclass
import sys
import io
import datetime
from contextlib import contextmanager

LOG_FILE_NAME = f"lab4_log.txt"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# Простые стилизаторы вывода
def hr(char='─', width=70):
    print(char * width)


def title(text):
    hr('═')
    print(f"  {text}")
    hr('═')


def section(text):
    print()
    hr('─')
    print(f"• {text}")
    hr('─')


def sub(text):
    print(f"  > {text}")


def rule(lhs, rhs_list):
    arrow = " → "
    for rhs in rhs_list:
        r = ' '.join(rhs) if isinstance(rhs, (list, tuple)) else (rhs if rhs else 'ε')
        print(f"    {lhs}{arrow}{r}")


def format_comparison_result(is_equivalent, grammar1_name, grammar2_name, method_description=""):
    """
    Форматирует результат сравнения грамматик в красивом виде
    """
    hr('·', 50)
    if is_equivalent:
        print(f"✓ ГРАММАТИКИ ЭКВИВАЛЕНТНЫ")
        print(f"  {grammar1_name} ≡ {grammar2_name}")
        if method_description:
            print(f"  Метод: {method_description}")
        print("  Языки, порождаемые грамматиками, совпадают")
    else:
        print(f"✗ ГРАММАТИКИ НЕ ЭКВИВАЛЕНТНЫ")
        print(f"  {grammar1_name} ≢ {grammar2_name}")
        if method_description:
            print(f"  Метод: {method_description}")
        print("  Языки, порождаемые грамматиками, различаются")
    hr('·', 50)
    print()


def compare_and_display(g1, g2, g1_name, g2_name, method_description="", max_length=5):
    """
    Сравнивает грамматики и выводит красиво отформатированный результат
    """
    result = g1.compare_generated_strings(g2, max_length=max_length)
    format_comparison_result(result, g1_name, g2_name, method_description)
    return result


@contextmanager
def console_logger(log_file_name=LOG_FILE_NAME):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_name, 'w', encoding='utf-8') as f:
        tee_out = Tee(original_stdout, f)
        tee_err = Tee(original_stderr, f)
        sys.stdout = tee_out
        sys.stderr = tee_err
        try:
            print(f"[LOG START] {datetime.datetime.now().isoformat(timespec='seconds')}")
            print(f"[LOG FILE]  {log_file_name}")
            hr()
            yield
        finally:
            hr()
            print(f"[LOG END]   {datetime.datetime.now().isoformat(timespec='seconds')}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr


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


def get_max_depth(node, depth=0):
    """Вычисляет максимальную глубину дерева."""
    if not node.children:
        return depth
    return max(get_max_depth(ch, depth + 1) for ch in node.children)


def print_tree_ascii_aligned(root):
    """Печатает дерево с выравниванием всех листьев по одному уровню."""

    max_depth = get_max_depth(root)

    def _print(node, prefix, is_last, current_depth):
        connector = "└─" if is_last else "├─"

        # Добавляем горизонтальные линии только для листьев
        if not node.children:
            # Количество линий зависит от разницы между макс. глубиной и текущей
            padding = (max_depth - current_depth) * 2
            horizontal_line = "─" * padding
        else:
            horizontal_line = ""

        print(prefix + connector + horizontal_line + node.sym)

        new_prefix = prefix + ("  " if is_last else "│ ")
        for i, ch in enumerate(node.children):
            _print(ch, new_prefix, i == len(node.children) - 1, current_depth + 1)

    # Печатаем корень
    print(root.sym)

    for i, ch in enumerate(root.children):
        _print(ch, "", i == len(root.children) - 1, 1)


def reconstruct_derivation(g, rule_path, tokens):
    """
    Восстанавливает последовательность форм из пути правил.
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


def main():
    title("Решение задачи 9: Все левые выводы для цепочки abab")
    tokens_9 = ['a', 'b', 'a', 'b']

    nonterminals_9 = {'S'}
    terminals_9 = {'a', 'b', 'ε'}
    productions_9 = {
        'S': [['a', 'S', 'b', 'S'], ['b', 'S', 'a', 'S'], []]
    }
    start_symbol_9 = 'S'
    g9 = Grammar(nonterminals_9, terminals_9, productions_9, start_symbol_9)

    section("Левые выводы и деревья")
    derivations = g9.all_left_derivations_deterministic(tokens_9, time_limit_s=10)
    for idx, rule_path in enumerate(derivations, 1):
        sub(f"Вывод {idx}")
        forms = reconstruct_derivation(g9, rule_path, tokens_9)
        print("  Формы: " + " => ".join(forms))
        print("  Дерево:")
        tree = build_tree_from_left_derivation(g9, rule_path)
        print_tree_ascii_aligned(tree)
        print()

    # Праволинейная грамматика для варианта a
    section("Праволинейная грамматика: вариант a")
    g_right_a = Grammar(
        nonterminals={'S', 'B', 'C'},
        terminals={'0', '1'},
        productions={
            'S': [['0', 'S'], ['0', 'B']],
            'B': [['1', 'B'], ['1', 'C']],
            'C': [['1', 'C'], []]
        },
        start_symbol='S'
    )
    g_right_a.display_info()
    # Праволинейная грамматика для варианта b
    section("Праволинейная грамматика: вариант b")
    g_right_b = Grammar(
        nonterminals={'S', 'A', 'B'},
        terminals={'a', 'b'},
        productions={
            'S': [['a', 'A'], ['a', 'B'], ['b', 'A']],
            'A': [['b', 'S']],
            'B': [['a', 'S'], ['b', 'B'], []]
        },
        start_symbol='S'
    )
    g_right_b.display_info()
    section("Леволинейные эквиваленты (детерминированные)")
    g_left_a = g_right_a.to_left_linear_deterministic()
    g_left_b = g_right_b.to_left_linear_deterministic()
    sub("Вариант a")
    g_left_a.display_info()
    sub("Вариант b")
    g_left_b.display_info()

    section("Сравнение порождаемых множеств (до длины 5)")
    compare_and_display(g_right_a, g_left_a, "Праволинейная_a", "Леволинейная_a",
                        "Сравнение строк до длины 5", max_length=5)

    compare_and_display(g_right_b, g_left_b, "Праволинейная_b", "Леволинейная_b",
                        "Сравнение строк до длины 5", max_length=5)

    section("G1 и G2: сравнение")

    # =========================
    # ПРОСТОЙ ПРИМЕР: ПРАВОЛИНЕЙНЫЕ
    # =========================
    G1r_simple = Grammar(
        nonterminals={'S', 'A'},
        terminals={'0', '1'},
        productions={
            'S': [['0', 'S'], ['1', 'A']],
            'A': [['0', 'A'], []]
        },
        start_symbol='S'
    )

    G2r_simple = Grammar(
        nonterminals={'S', 'B'},
        terminals={'0', '1'},
        productions={
            'S': [['0', 'S'], ['1', 'B']],
            'B': [['1', 'B'], []]
        },
        start_symbol='S'
    )

    sub("Пересечение (праволинейные, тестовые)")
    G_inter_r, dfa_inter_r = G1r_simple.intersection_preserving_names(G2r_simple)
    G_inter_r.display_info()

    sub("Порождаемые строки пересечения (праволинейные, длина до 10)")
    strings_r = G_inter_r.generate_strings(10)
    print(f"  Количество строк: {len(strings_r)}")
    for length in range(11):
        length_strings = [s for s in strings_r if len(s) == length and s != '']
        if length_strings:
            print(f"  Длина {length}: {', '.join(sorted(length_strings))}")

    # =========================
    # ПРОСТОЙ ПРИМЕР: ЛЕВОЛИНЕЙНЫЕ
    # =========================
    G1l_simple = Grammar(
        nonterminals={'S', 'A'},
        terminals={'0', '1'},
        productions={
            'S': [['S', '0'], ['A', '1']],
            'A': [['A', '0'], []]
        },
        start_symbol='S'
    )

    G2l_simple = Grammar(
        nonterminals={'S', 'B'},
        terminals={'0', '1'},
        productions={
            'S': [['S', '0'], ['B', '1']],
            'B': [['B', '1'], []]
        },
        start_symbol='S'
    )

    sub("Пересечение (леволинейные, тестовые)")
    G_inter_l, dfa_inter_l = G1l_simple.intersection_preserving_names(G2l_simple)
    G_inter_l.display_info()

    sub("Порождаемые строки пересечения (леволинейные, длина до 10)")
    strings_l = G_inter_l.generate_strings(10)
    print(f"  Количество строк: {len(strings_l)}")
    for length in range(11):
        length_strings = [s for s in strings_l if len(s) == length and s != '']
        if length_strings:
            print(f"  Длина {length}: {', '.join(sorted(length_strings))}")

    # =========================
    # ЛЕВОЛИНЕЙНЫЕ, ПОЛУЧЕННЫЕ ИЗ ПРАВОЛИНЕЙНЫХ
    # =========================
    sub("Пересечение (леволинейные, полученные из праволинейных тестовых)")

    # 1) Берем тестовые праволинейные
    G1r_base = Grammar(
        nonterminals={'S', 'A'},
        terminals={'0', '1'},
        productions={
            'S': [['0', 'S'], ['1', 'A']],
            'A': [['0', 'A'], []]
        },
        start_symbol='S'
    )

    G2r_base = Grammar(
        nonterminals={'S', 'B'},
        terminals={'0', '1'},
        productions={
            'S': [['0', 'S'], ['1', 'B']],
            'B': [['1', 'B'], []]
        },
        start_symbol='S'
    )

    # 2) Преобразуем каждую в леволинейную форму

    G1_from_right_to_left = G1r_base.to_left_linear_deterministic()
    G2_from_right_to_left = G2r_base.to_left_linear_deterministic()

    # 3) Строим пересечение уже для леволинейных, полученных из праволинейных
    G_inter_from_right_left, dfa_inter_from_right_left = \
        G1_from_right_to_left.intersection_preserving_names(G2_from_right_to_left)

    # 4) Выводим информацию
    section("Инфо: пересечение (леволинейные ← праволинейные тестовые)")
    G_inter_from_right_left.display_info()

    sub("Порождаемые строки (леволинейные ← праволинейные тестовые, длина до 10)")
    strings_from_right_left = G_inter_from_right_left.generate_strings(10)
    print(f"  Количество строк: {len(strings_from_right_left)}")
    for length in range(11):
        length_strings = [s for s in strings_from_right_left if len(s) == length and s != '']
        if length_strings:
            print(f"  Длина {length}: {', '.join(sorted(length_strings))}")

    # =========================
    # ПО ЗАДАНИЮ (леволинейные)
    # =========================
    G1_task = Grammar(
        nonterminals={'S', 'A'},
        terminals={'0', '1'},
        productions={
            'S': [['S', '1'], ['A', '0']],
            'A': [['A', '1'], ['0']]
        },
        start_symbol='S'
    )

    G2_task = Grammar(
        nonterminals={'S', 'A', 'B', 'C', 'D', 'E'},
        terminals={'0', '1'},
        productions={
            'S': [['A', '1'], ['B', '0'], ['E', '1']],
            'A': [['S', '1']],
            'B': [['C', '1'], ['D', '1']],
            'C': [['0']],
            'D': [['B', '1']],
            'E': [['E', '0'], ['1']]
        },
        start_symbol='S'
    )

    sub("Пересечение (леволинейные, по заданию)")
    G_inter_task, dfa_inter_task = G1_task.intersection_preserving_names(G2_task)
    G_inter_task.display_info()

    sub("Порождаемые строки пересечения (по заданию, длина до 10)")
    strings_task = G_inter_task.generate_strings(10)
    print(f"  Количество строк: {len(strings_task)}")
    for length in range(11):
        length_strings = [s for s in strings_task if len(s) == length and s != '']
        if length_strings:
            print(f"  Длина {length}: {', '.join(sorted(length_strings))}")


if __name__ == "__main__":
    with console_logger():
        main()