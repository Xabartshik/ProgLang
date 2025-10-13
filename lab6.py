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