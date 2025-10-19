from collections import defaultdict, deque
from lib.CFGParser import Grammar, StateDiagram
from dataclasses import dataclass
import sys
import io
import datetime
from contextlib import contextmanager

LOGFILENAME = "lab7_log.txt"

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

@contextmanager
def tee_stdout(*streams):
    old_stdout = sys.stdout
    try:
        sys.stdout = Tee(*streams)
        yield
    finally:
        sys.stdout = old_stdout

def hr(char="=", width=70):
    print(char * width)

def title(text):
    hr()
    print(f" {text}")
    hr()

def section(text):
    print()
    hr(".", 50)
    print(f" {text}")
    hr(".", 50)

def sub(text):
    print(f"  {text}")

@dataclass
class Node:
    sym: str
    children: list

def build_tree_from_left_derivation(g, rule_path):
    """Строит дерево разбора из левого вывода."""
    root = Node(g.start_symbol, [])
    form_syms = [g.start_symbol]
    form_nodes = [root]

    for nt, prod in rule_path:
        left_idx = next(i for i, s in enumerate(form_syms) if s in g.nonterminals)
        assert form_syms[left_idx] == nt, "Некорректный путь левого вывода"

        if len(prod) == 0:
            form_nodes[left_idx].children = [Node('ε', [])]
            form_syms = form_syms[:left_idx] + form_syms[left_idx+1:]
            form_nodes = form_nodes[:left_idx] + form_nodes[left_idx+1:]
        else:
            child_nodes = [Node(s, []) for s in prod]
            form_nodes[left_idx].children = child_nodes
            form_syms = form_syms[:left_idx] + prod + form_syms[left_idx+1:]
            form_nodes = form_nodes[:left_idx] + child_nodes + form_nodes[left_idx+1:]

    return root

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

        if not node.children:
            padding = (max_depth - current_depth) * 2
            horizontal_line = "─" * padding
        else:
            horizontal_line = ""

        print(prefix + connector + horizontal_line + node.sym)
        new_prefix = prefix + ("  " if is_last else "│ ")

        for i, ch in enumerate(node.children):
            _print(ch, new_prefix, i == len(node.children) - 1, current_depth + 1)

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
    forms.append(''.join(form))

    for nt, prod in rule_path:
        left_idx = next((i for i, sym in enumerate(form) if sym in g.nonterminals), None)
        if left_idx is None or form[left_idx] != nt:
            raise ValueError("Некорректный путь вывода")

        form = form[:left_idx] + prod + form[left_idx + 1:]
        forms.append(''.join(form))

    if form != tokens:
        raise ValueError("Вывод не генерирует заданную цепочку")

    return forms

def main():
    with open(LOGFILENAME, "w", encoding="utf-8") as logfile:
        with tee_stdout(sys.stdout, logfile):
            title("ЛАБОРАТОРНАЯ РАБОТА №7 — Анализ грамматики и построение ДС")
            print(f"Дата выполнения: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            section("Исходная грамматика")
            print("Дана грамматика:")
            print("  S → Sb | Aa | a | b")
            print("  A → Aa | Sb | a")
            print()

            # Определяем грамматику
            nonterminals = {'S', 'A'}
            terminals = {'a', 'b'}
            productions = {
                'S': [['S', 'b'], ['A', 'a'], ['a'], ['b']],
                'A': [['A', 'a'], ['S', 'b'], ['a']]
            }
            start_symbol = 'S'

            g = Grammar(nonterminals, terminals, productions, start_symbol)

            print("Информация о грамматике:")
            g.display_info()


            section("Построение диаграммы состояний")

            ds = g.build_state_diagram_nfa()

            print("Полученная диаграмма состояний:")
            ds.display()

            # Рисуем диаграмму состояний
            ds.draw_state_diagram_svg()

            section("Тестирование цепочек")
            test_strings = [
                ['a'],  # 'a'
                ['b'],  # 'b'
                ['a', 'a'],  # 'aa'
                ['a', 'b'],  # 'ab'
                ['b', 'a'],  # 'ba'
                ['b', 'b'],  # 'bb'
                ['a', 'a', 'a'],  # 'aaa'
                ['a', 'b', 'a'],  # 'aba'
                ['b', 'b', 'b'],  # 'bbb'
                ['a', 'b', 'a', 'b'],  # 'abab'
            ]

            parsed_strings = []

            for test_str in test_strings:
                print(f"\nРазбор цепочки '{test_str}':")
                result = g.parse_nfa(test_str)

                if result.get("accepted"):
                    count = result.get("num_paths", len(result.get("paths", [])))
                    print(f"  ✓ Цепочка '{test_str}' ПРИНИМАЕТСЯ ({count} пут{'ь' if count == 1 else 'ей'})")
                    # Вывести все найденные пути
                    for i, path in enumerate(result["paths"], start=1):
                        # Преобразуем элементы пути в строки
                        formatted = []
                        for element in path:
                            if isinstance(element, tuple):
                                # например ('a', 'S') → "a→S"
                                formatted.append(f"{element[0]}→{element[1]}")
                            else:
                                formatted.append(str(element))
                        print(f"    Путь {i}: {' → '.join(formatted)}")
                    parsed_strings.append(test_str)
                else:
                    reason = result.get("reason", "неизвестная причина")
                    print(f"  ✗ Цепочка '{test_str}' ОТКЛОНЯЕТСЯ ({reason})")

            section("Генерация цепочек языка  L(G) = { w ∈ {a^n,b^m}, n => 0, m => m")
            print("Генерируем цепочки до длины 6:")
            print()

            generated_strings = g.generate_strings(6)

            by_length = defaultdict(list)
            for s in generated_strings:
                if s == '':
                    by_length[0].append("ε")
                else:
                    by_length[len(s)].append(s)

            for length in sorted(by_length.keys()):
                if by_length[length]:
                    shown = sorted(by_length[length])[:30]
                    print(f"Длина {length}: {' '.join(shown)}")
                    if len(by_length[length]) > 30:
                        print(f"  ... и ещё {len(by_length[length]) - 30} цепочек")

            section("Деревья вывода для тестовых цепочек")

            test_tokens = [
                ['a'],
                ['b'],
                ['a', 'b'],
                ['b', 'a'],
                ['a', 'a', 'a'],
                ['a', 'b', 'a', 'b']
            ]

            for test_idx, tokens in enumerate(test_tokens, 1):
                test_str = ''.join(tokens)
                sub(f"Строка {test_idx}: {test_str}")
                try:
                    derivations = g.all_left_derivations_deterministic(tokens)
                    if not derivations:
                        print("  Строка не принадлежит языку грамматики.")
                        print()
                        continue

                    for idx, rule_path in enumerate(derivations[:3], 1):
                        if len(derivations) > 1:
                            sub(f"  Вывод {idx}")

                        forms = reconstruct_derivation(g, rule_path, tokens)
                        print(f"    Формы: {' => '.join(forms)}")
                        print("    Дерево:")
                        tree = build_tree_from_left_derivation(g, rule_path)
                        print_tree_ascii_aligned(tree)
                        print()

                    if len(derivations) > 3:
                        print(f"    ... и ещё {len(derivations) - 3} вариантов вывода")
                        print()

                except Exception as e:
                    print(f"  Ошибка при разборе: {e}")
                    print()

            print()
            hr()
            print(f"Логирование завершено. Результаты сохранены в {LOGFILENAME}")
            hr()

if __name__ == "__main__":
    main()
