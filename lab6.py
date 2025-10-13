from collections import defaultdict, deque
from lib.CFGParser import Grammar, StateDiagram
from dataclasses import dataclass
import sys
import io
import datetime
from contextlib import contextmanager

LOGFILENAME = "lab6_log.txt"

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
    print(f"   {text}")


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
    with open(LOGFILENAME, "w", encoding="utf-8") as logfile:
        with tee_stdout(sys.stdout, logfile):
            title("ЛАБОРАТОРНАЯ РАБОТА №6 — Задание 2.2 (ДС → грамматика)")
            print(f"Дата выполнения: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            section("Диаграмма состояний (задание 2)")
            ds = StateDiagram()
            for s in ["H", "A", "B", "S", "ER"]: ds.add_state(s)
            ds.set_start_state("H")
            ds.add_final_state("S")

            # Основные допустимые переходы (0/1 и конец ⊥)
            for t in ("0", "1"): ds.add_transition("H", t, "A")
            for t in ("0", "1"): ds.add_transition("A", t, "A")
            for t in ("0", "1"): ds.add_transition("B", t, "A")
            for t in ("+", "-"): ds.add_transition("A", t, "B")
            ds.add_transition("A", "⊥", "S")


            for u in ["H", "A", "B"]:
                ds.add_transition(u, "ㅤ", "ER")
            ds.display()
            ds.draw_state_diagram_svg()

            section("Разбор требуемых цепочек")
            tests = ["1011⊥", "10+011⊥", "0-101+1⊥"]
            parsed = []
            for test_str in tests:
                print(f"\nРазбор цепочки '{test_str}':")
                result = ds.parse(test_str)
                if result:
                    print(f"  ✓ Цепочка '{test_str}' ПРИНИМАЕТСЯ")
                    print(f"    Путь разбора: {' → '.join(result['path'])}")
                    parsed.append(test_str)
                else:
                    print(f"  ✗ Цепочка '{test_str}' ОТКЛОНЯЕТСЯ")

            section("Восстановление леволинейной грамматики из ДС")
            g = ds.to_left_linear_grammar(end_symbol="⊥", error_state="ER", start_nonterminal="S")
            g.display_info()
            print("Деревья вывода:")
            tests_tokenized = [
                ['1', '0', '1', '1', '⊥'],  # "1011⊥"
                ['1', '0', '+', '0', '1', '1', '⊥'],  # "10+011⊥"
                ['0', '-', '1', '0', '1', '+', '1', '⊥']  # "0-101+1⊥"
            ]
            for test_idx, tokens in enumerate(tests_tokenized, 1):
                sub(f"Строка {test_idx}: {''.join(tokens)}")
                try:
                    derivations = g.all_left_derivations_deterministic(tokens)

                    if not derivations:
                        print("  Строка не принадлежит языку грамматики.")
                        print()
                        continue

                    # Вывод всех деревьев разбора
                    for idx, rule_path in enumerate(derivations, 1):
                        sub(f"  Вывод {idx}")
                        forms = reconstruct_derivation(g, rule_path, tokens)
                        print("    Формы: " + " => ".join(forms))
                        print("    Дерево:")
                        tree = build_tree_from_left_derivation(g, rule_path)
                        print_tree_ascii_aligned(tree)
                        print()
                except Exception as e:
                    print(f"  Ошибка при разборе: {e}")
                    print()
            print("Генерация цепочек языка (до длины 8):")
            generated_strings = g.generate_strings(8)
            gen_set = set(generated_strings)
            by_length = defaultdict(list)
            for s in generated_strings:
                by_length[len(s)].append("ε" if s == "" else s)
            for length in sorted(by_length.keys()):
                if by_length[length]:
                    shown = sorted(by_length[length])[:20]
                    print(f"Длина {length}: {' '.join(shown)}")
                    if len(by_length[length]) > 20:
                        print(f"           ... и ещё {len(by_length[length]) - 20} цепочек")
                    matches = sorted({("ε" if p == "" else p) for p in parsed if p in gen_set and len(p) == length})
                    if matches:
                        print(f"           Обнаружены разобранные ранее строки: {' '.join(matches)}")

if __name__ == "__main__":
    main()
    print("Больно дюже работает?")