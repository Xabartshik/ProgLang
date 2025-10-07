from collections import defaultdict, deque
from lib.CFGParser import Grammar
from dataclasses import dataclass
import sys
import io
import datetime
from contextlib import contextmanager

LOGFILENAME = "lab5_log.txt"

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

def rule(lhs, rhslist, arrow=" -> "):
    for rhs in rhslist:
        r = " ".join(rhs) if isinstance(rhs, (list, tuple)) else rhs
        if not rhs:
            r = "ε"
        print(f"      {lhs}{arrow}{r}")

def format_comparison_result(isequivalent, grammar1name, grammar2name, methoddescription=""):
    hr("-", 50)
    if isequivalent:
        print("✓ Эквивалентны ✓")
        print(f"  {grammar1name} ≡ {grammar2name}")
        if methoddescription:
            print(f"  ({methoddescription})")
        print("   🎉 ", "🎉 ", "🎉 ")
    else:
        print("✗ НЕ эквивалентны ✗")
        print(f"  {grammar1name} ≢ {grammar2name}")
        if methoddescription:
            print(f"  ({methoddescription})")
        print("   ❌ ", "❌ ", "❌ ")
    hr("-", 50)
    print()

def compare_and_display(g1, g2, g1name, g2name, description="", maxlength=5):
    strings1 = g1.generate_strings(maxlength)
    strings2 = g2.generate_strings(maxlength)

    isequivalent = (set(strings1) == set(strings2))

    print(f"Сравнение языков: {g1name} vs {g2name}")
    if description:
        print(f"Метод сравнения: {description}")

    print(f"\nЦепочки из L({g1name}) до длины {maxlength}:")
    all_lengths1 = defaultdict(list)
    for s in strings1:
        if s == '':
            all_lengths1[0].append("ε")
        else:
            all_lengths1[len(s)].append(s)

    for length in range(maxlength + 1):
        if length in all_lengths1:
            print(f"Длина {length}: {' '.join(sorted(all_lengths1[length]))}")

    print(f"\nЦепочки из L({g2name}) до длины {maxlength}:")
    all_lengths2 = defaultdict(list)
    for s in strings2:
        if s == '':
            all_lengths2[0].append("ε")
        else:
            all_lengths2[len(s)].append(s)

    for length in range(maxlength + 1):
        if length in all_lengths2:
            print(f"Длина {length}: {' '.join(sorted(all_lengths2[length]))}")

    format_comparison_result(isequivalent, g1name, g2name, description)

def main():
    with open(LOGFILENAME, 'w', encoding='utf-8') as logfile:
        with tee_stdout(sys.stdout, logfile):
            section("ЗАДАНИЕ 1: Построение диаграммы состояний для регулярной грамматики")

            print("Дана регулярная грамматика с правилами:")
            print()
            print("S → S0 | S1 | P0 | P1")
            print("P → N.")
            print("N → 0 | 1 | N0 | N1")
            print()

            # Определяем грамматику
            nonterminals = {'S', 'P', 'N'}
            terminals = {'0', '1', '.'}
            productions = {
                'S': [['S', '0'], ['S', '1'], ['P', '0'], ['P', '1']],
                'P': [['N', '.']],
                'N': [['0'], ['1'], ['N', '0'], ['N', '1']]
            }
            startsymbol = 'S'

            g = Grammar(nonterminals, terminals, productions, startsymbol)

            print("Информация о грамматике:")
            g.display_info()

            print("\nПроверка типа грамматики:")
            print(f"  - Является ли леволинейной: {g.is_left_linear()}")
            print(f"  - Является ли праволинейной: {g.is_right_linear()}")
            print(f"  - Является ли регулярной: {g.is_regular()}")

            # Построение диаграммы состояний
            section("Построение диаграммы состояний")
            print("Построение диаграммы состояний по алгоритму из методического пособия:")
            print()
            print("1. Создаём состояния:")
            print("   - H (начальное состояние)")
            print("   - S, P, N (по одному для каждого нетерминала)")
            print()
            print("2. Строим переходы:")
            print("   - Для правил вида W → t: дуга H → W с меткой t")
            print("   - Для правил вида W → Vt: дуга V → W с меткой t")
            print()

            ds = g.build_state_diagram()

            print("Полученная диаграмма состояний:")
            ds.display()
            ds.draw_state_diagram_svg()
            # Тестирование цепочек
            section("Разбор тестовых цепочек")

            test_strings = ["11.010", "0.1", "01.", "100"]

            for test_str in test_strings:
                print(f"\nРазбор цепочки '{test_str}':")
                result = ds.parse(test_str)
                if result:
                    print(f"  ✓ Цепочка '{test_str}' ПРИНИМАЕТСЯ")
                    print(f"    Путь разбора: {' → '.join(result['path'])}")
                else:
                    print(f"  ✗ Цепочка '{test_str}' ОТКЛОНЯЕТСЯ")

            # Анализ языка
            section("Анализ порождаемого языка")

            print("Генерация цепочек языка (до длины 8):")
            generated_strings = g.generate_strings(8)

            # Группируем по длинам
            by_length = defaultdict(list)
            for s in generated_strings:
                if s == '':
                    by_length[0].append("ε")
                else:
                    by_length[len(s)].append(s)

            for length in sorted(by_length.keys()):
                if by_length[length]:
                    print(f"Длина {length}: {' '.join(sorted(by_length[length])[:20])}")
                    if len(by_length[length]) > 20:
                        print(f"           ... и ещё {len(by_length[length])-20} цепочек")

            print("\nОписание языка:")
            print("Данная грамматика порождает язык вещественных чисел в десятичной записи.")
            print("Язык L(G) = {цепочки вида d₁d₂...dₙ.d_{n+1}d_{n+2}...d_m, где dᵢ ∈ {0,1}}")
            print("То есть двоичные числа с обязательной десятичной точкой.")
            print("Примеры: 0.1, 11.010, 101.001, 1.0, и т.д.")
            print("\nЛогирование завершено. Результаты сохранены в", LOGFILENAME)

if __name__ == "__main__":
    main()
