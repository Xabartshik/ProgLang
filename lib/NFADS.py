from collections import defaultdict
from typing import Dict, Tuple, Set


# Переписанные методы для класса Grammar с поддержкой NFA (недетерминированных автоматов)

def build_state_diagram_nfa(self):
    """
    Построить недетерминированную диаграмму состояний для леволинейной грамматики.

    Алгоритм (леволинейная: W → t | W → V t):
    1) Добавляем состояния для всех нетерминалов + специальное начальное состояние H
    2) Для W → t добавляем дугу H --t--> W (может быть несколько для одного t)
    3) Для W → V t добавляем дугу V --t--> W (может быть несколько переходов)
    4) Стартовый символ грамматики делаем конечным состоянием

    Поддерживает недетерминированность: из одного состояния по одному символу
    могут быть переходы в несколько различных состояний.
    """
    if not self.is_left_linear():
        raise ValueError("Диаграмма состояний строится только для леволинейных грамматик")

    ds = StateDiagramNFA()  # Используем недетерминированную версию

    # 1) Состояния: H и все нетерминалы
    H = 'H'
    ds.set_start_state(H)
    for nt in sorted(self.nonterminals):
        ds.add_state(nt)

    # 4) Стартовый символ — конечное состояние
    ds.add_final_state(self.start_symbol)

    # 2) и 3) Переходы (теперь множественные)
    for W, rhslist in self.productions.items():
        for rhs in rhslist:
            # ε-продукций здесь нет, но оставим задел
            if len(rhs) == 0:
                # При наличии W → ε обычно отмечают W как конечное
                if W == self.start_symbol:
                    ds.add_final_state(W)
                continue

            # W → t
            if len(rhs) == 1 and rhs[0] in self.terminals:
                t = rhs[0]
                # Дуга H --t--> W (может быть недетерминизм!)
                ds.add_transition(H, t, W)
                continue

            # W → V t
            if len(rhs) == 2 and rhs[0] in self.nonterminals and rhs[1] in self.terminals:
                V, t = rhs[0], rhs[1]
                # Дуга V --t--> W (может быть недетерминизм!)
                ds.add_transition(V, t, W)
                continue

            # Если попалось что-то иное — это не леволинейная форма
            raise ValueError(f"Недопустимое леволинейное правило: {W} → {''.join(rhs)}")

    return ds


def parse_nfa(self, input_string):
    """
    Разбор строки по недетерминированной диаграмме состояний.
    Использует алгоритм обхода в ширину для исследования всех возможных путей.

    Возвращает:
    - accepted: True если хотя бы один путь приводит к финальному состоянию
    - paths: список всех успешных путей (если найдены)
    - reason: причина отклонения (если не принято)
    """
    if not self.is_left_linear():
        raise ValueError("Разбор доступен только для леволинейных грамматик")

    if not input_string:
        return {"accepted": False, "reason": "Пустая строка"}

    # Строим NFA диаграмму
    nfa = self.build_state_diagram_nfa()

    # Используем BFS для обхода всех возможных путей
    from collections import deque

    # Очередь: (текущая позиция в строке, текущее состояние, путь)
    queue = deque([(0, nfa.start_state, [nfa.start_state])])
    visited = set()  # (позиция, состояние)
    successful_paths = []

    # Шаг 1: обработка первого символа
    first_symbol = input_string[0]
    initial_states = nfa.transitions.get((nfa.start_state, first_symbol), set())

    if not initial_states:
        return {
            "accepted": False,
            "reason": f"Нет переходов из начального состояния по символу '{first_symbol}'"
        }

    # Инициализируем очередь всеми возможными начальными переходами
    queue = deque()
    for state in initial_states:
        queue.append((1, state, [nfa.start_state, (first_symbol, state)]))

    # Обрабатываем остальные символы
    while queue:
        pos, current_state, path = queue.popleft()

        # Избегаем повторной обработки одного и того же состояния на той же позиции
        state_key = (pos, current_state)
        if state_key in visited:
            continue
        visited.add(state_key)

        # Если обработали всю строку
        if pos == len(input_string):
            # Проверяем, является ли текущее состояние финальным
            if current_state in nfa.final_states:
                successful_paths.append(path[:])
            continue

        # Берем следующий символ
        next_symbol = input_string[pos]

        # Получаем все возможные переходы из текущего состояния
        next_states = nfa.transitions.get((current_state, next_symbol), set())

        # Добавляем все возможные переходы в очередь
        for next_state in next_states:
            new_path = path + [(next_symbol, next_state)]
            queue.append((pos + 1, next_state, new_path))

    if successful_paths:
        return {
            "accepted": True,
            "paths": successful_paths,
            "num_paths": len(successful_paths)
        }
    else:
        return {
            "accepted": False,
            "reason": "Ни один путь не привел к финальному состоянию"
        }


# Новый класс для недетерминированной диаграммы состояний
class StateDiagramNFA:
    """Класс для представления недетерминированной диаграммы состояний (NFA)"""

    def __init__(self):
        self.states = set()
        self.start_state = None
        self.final_states = set()
        # Ключевое отличие: transitions теперь отображает (state, symbol) -> SET of states
        self.transitions: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add_state(self, state):
        """Добавить состояние"""
        self.states.add(state)

    def set_start_state(self, state):
        """Установить начальное состояние"""
        self.start_state = state
        self.states.add(state)

    def add_final_state(self, state):
        """Добавить финальное состояние"""
        self.final_states.add(state)
        self.states.add(state)

    def add_transition(self, from_state, symbol, to_state):
        """
        Добавить переход. В отличие от DFA, может быть несколько переходов
        из одного состояния по одному символу.
        """
        self.states.add(from_state)
        self.states.add(to_state)
        # Добавляем в множество (поддержка недетерминизма)
        self.transitions[(from_state, symbol)].add(to_state)

    def display(self):
        """Вывести диаграмму состояний"""
        print("Недетерминированная диаграмма состояний (NFA):")
        print(f"Состояния: {{{', '.join(sorted(self.states))}}}")
        print(f"Начальное состояние: {self.start_state}")
        print(f"Финальные состояния: {{{', '.join(sorted(self.final_states))}}}")
        print("Переходы:")

        # Группируем переходы по (from_state, symbol)
        for (from_state, symbol), to_states in sorted(self.transitions.items()):
            # Если несколько целевых состояний - показываем недетерминизм
            if len(to_states) > 1:
                to_states_str = "{" + ", ".join(sorted(to_states)) + "}"
                print(f"  {from_state} --({symbol})--> {to_states_str} [НЕДЕТЕРМИНИЗМ]")
            else:
                to_state = list(to_states)[0]
                print(f"  {from_state} --({symbol})--> {to_state}")

    def display_ascii(self):
        """
        Компактная ASCII-визуализация с указанием недетерминированных переходов.
        """
        print("Недетерминированная диаграмма состояний (ASCII):")
        print(f"Состояния: {{{', '.join(sorted(self.states))}}}")
        print(f"Начальное: {self.start_state}")
        print(f"Финальные: {{{', '.join(sorted(self.final_states))}}}")

        # Группируем: (from, to) -> set of symbols
        grouped = {}
        for (s, a), targets in self.transitions.items():
            for t in targets:
                grouped.setdefault((s, t), set()).add(a)

        # Вывод с пометками недетерминизма
        print("Переходы:")
        for (s, t), labels in sorted(grouped.items()):
            lab = ",".join(sorted(labels, key=str))

            # Проверяем, есть ли недетерминизм (несколько целей для одного символа)
            is_nondet = False
            for symbol in labels:
                if len(self.transitions.get((s, symbol), set())) > 1:
                    is_nondet = True
                    break

            if is_nondet:
                print(f"  {s} --({lab})--> {t} [*]")
            else:
                print(f"  {s} --({lab})--> {t}")

        print("\n[*] - недетерминированный переход")

    def draw_state_diagram_svg(self, filename="state_diagram_nfa.svg", layout_hints=None):
        """
        Визуализация недетерминированной диаграммы состояний через svgwrite (SVG).
        Недетерминированные переходы выделяются визуально.
        """
        import math
        import svgwrite

        layout_hints = layout_hints or {}

        # ---------- утилиты ----------
        def norm_label(sym):
            s = ("" if sym is None else str(sym)).strip()
            return None if s in {"", "ε", "eps", "epsilon"} else s

        # Группировка меток и петель
        grouped, loops = {}, {}
        nondet_edges = set()  # Набор недетерминированных рёбер

        # Сначала определяем, какие переходы недетерминированные
        for (u, a), targets in self.transitions.items():
            na = norm_label(a)
            if len(targets) > 1:
                # Недетерминизм: из u по a идут переходы в несколько состояний
                for v in targets:
                    nondet_edges.add((u, v, na))

            for v in targets:
                na_norm = norm_label(a)
                if u == v:
                    if na_norm is not None:
                        loops.setdefault(u, set()).add(na_norm)
                    else:
                        loops.setdefault(u, set())
                else:
                    if na_norm is None:
                        continue
                    grouped.setdefault((u, v), set()).add(na_norm)

        states = sorted(self.states)

        # ---------- выбор «ядра» (взаимные рёбра) ----------
        def mutual_weight(x, y):
            return len(grouped.get((x, y), set())) + len(grouped.get((y, x), set()))

        core_pair, best_w = None, -1
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                w = mutual_weight(states[i], states[j])
                if w > best_w:
                    best_w, core_pair = w, (states[i], states[j]) if w > 0 else None

        # ---------- координаты в «условных единицах» ----------
        pos = {}
        def setp(name, xy):
            if name in self.states and name not in pos:
                pos[name] = xy

        start = self.start_state
        finals = sorted(s for s in self.final_states if s in self.states)
        er_name = next((s for s in self.states if s in {"ER", "ERR", "ERROR"}), None)
        s_name = next((s for s in finals if s in {"S", "Σ", "ACCEPT", "OK"}), finals[0] if finals else None)

        if start: setp(start, (-2.2, 0.0))
        if s_name: setp(s_name, (2.6, 0.0))
        if er_name: setp(er_name, (-1.4, -1.0))

        if core_pair:
            u, v = core_pair
            setp(u, (0.0, 0.9))
            setp(v, (0.0, -0.9))

        remaining = [s for s in states if s not in pos]
        x_slots = [-1.0, -0.3, 0.3, 1.0, 1.8]
        for i, s in enumerate(remaining):
            x = x_slots[min(i, len(x_slots) - 1)]
            y = 0.0
            pos[s] = (x, y)

        # Переопределения пользователя
        for k, xy in (layout_hints or {}).items():
            pos[k] = xy

        # ---------- масштаб и стиль ----------
        unit = 140.0
        R = 0.22 * unit
        pad = 1.2 * unit

        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        minx, maxx = min(xs) - 1.2, max(xs) + 1.2
        miny, maxy = min(ys) - 1.2, max(ys) + 1.2
        width = (maxx - minx) * unit + pad
        height = (maxy - miny) * unit + pad

        def U(x, y):
            return ((x - minx) * unit + pad / 2, height - ((y - miny) * unit + pad / 2))

        dwg = svgwrite.Drawing(filename=filename, size=(width, height), profile='full')

        # Маркеры стрелок (обычные и для недетерминированных)
        marker_arrow = dwg.marker(insert=(6, 3), size=(6, 6), orient="auto")
        marker_arrow.add(dwg.path(d="M0,0 L0,6 L6,3 z", fill="#546E7A"))
        dwg.defs.add(marker_arrow)

        # Красный маркер для недетерминированных переходов
        marker_arrow_nondet = dwg.marker(insert=(6, 3), size=(6, 6), orient="auto")
        marker_arrow_nondet.add(dwg.path(d="M0,0 L0,6 L6,3 z", fill="#D32F2F"))
        dwg.defs.add(marker_arrow_nondet)

        marker_start = dwg.marker(insert=(8, 4), size=(8, 8), orient="auto")
        marker_start.add(dwg.path(d="M0,0 L0,8 L8,4 z", fill="#2E7D32"))
        dwg.defs.add(marker_start)

        # ---------- узлы ----------
        for s in states:
            x, y = U(*pos[s])
            face, edge, lw = "#E3F2FD", "#1976D2", 3
            if s == start:
                face, edge, lw = "#C8E6C9", "#2E7D32", 4
            if s in self.final_states:
                face, edge, lw = "#FFECB3", "#F57C00", 4

            dwg.add(dwg.circle(center=(x, y), r=R, fill=face, stroke=edge, stroke_width=lw))
            if s in self.final_states:
                dwg.add(dwg.circle(center=(x, y), r=R * 0.83, fill="none", stroke=edge, stroke_width=3))
            dwg.add(dwg.text(str(s), insert=(x, y + 5), text_anchor="middle",
                           font_size=18, font_weight="bold", fill="#263238"))

        # ---------- стартовая стрелка ----------
        if start in pos:
            sx, sy = pos[start]
            src = U(sx - 0.75, sy)
            dst = U(sx - 0.22, sy)
            start_line = dwg.line(start=src, end=dst, stroke="#2E7D32", stroke_width=4)
            start_line['marker-end'] = marker_start.get_funciri()
            dwg.add(start_line)
            tpos = U(sx - 1.0, sy)
            dwg.add(dwg.text("START", insert=tpos, text_anchor="middle",
                           font_size=14, font_weight="bold", fill="#2E7D32"))

        # ---------- рисование рёбер ----------
        def add_label(text, px, py, is_nondet=False):
            w = max(24, 9 * len(text) + 12)
            h = 22
            # Красный фон для недетерминированных переходов
            bg_color = "#FFCDD2" if is_nondet else "white"
            text_color = "#D32F2F" if is_nondet else "#D32F2F"

            dwg.add(dwg.rect(insert=(px - w / 2, py - h / 2), size=(w, h),
                           rx=6, ry=6, fill=bg_color, stroke="#BDBDBD", stroke_width=1))
            dwg.add(dwg.text(text, insert=(px, py + 5), text_anchor="middle",
                           font_size=13, font_weight="bold", fill=text_color))

        def draw_edge(u, v, label, rad=0.0, is_nondet=False):
            ux, uy = pos[u]
            vx, vy = pos[v]
            sx, sy = U(ux, uy)
            tx, ty = U(vx, vy)

            # Смещение до края окружности
            dx, dy = (tx - sx, ty - sy)
            L = math.hypot(dx, dy) or 1.0
            sx, sy = (sx + dx / L * R, sy + dy / L * R)
            tx, ty = (tx - dx / L * R, ty - dy / L * R)

            # Контрольная точка
            nx, ny = (-dy / L, dx / L)
            bend = 80 * rad
            cx, cy = ((sx + tx) / 2 + nx * bend, (sy + ty) / 2 + ny * bend)

            # Цвет для недетерминированных переходов
            stroke_color = "#D32F2F" if is_nondet else "#546E7A"
            stroke_width = 4 if is_nondet else 3

            path = dwg.path(d=f"M {sx},{sy} Q {cx},{cy} {tx},{ty}",
                          fill="none", stroke=stroke_color, stroke_width=stroke_width)
            marker = marker_arrow_nondet if is_nondet else marker_arrow
            path['marker-end'] = marker.get_funciri()
            dwg.add(path)

            # Метка
            lx, ly = ((sx + tx) / 2 + nx * (bend + 18), (sy + ty) / 2 + ny * (bend + 18))
            add_label(label, lx, ly, is_nondet=is_nondet)

        # Петли
        for s, labels in loops.items():
            if s not in pos:
                continue
            x, y = pos[s]
            cx, cy = U(x, y)
            rr = R * 0.95

            # Проверяем недетерминизм в петлях
            is_nondet_loop = any(
                len(self.transitions.get((s, label), set())) > 1
                for label in labels
            )

            stroke_color = "#D32F2F" if is_nondet_loop else "#546E7A"
            marker = marker_arrow_nondet if is_nondet_loop else marker_arrow

            loop_path = dwg.path(d=(
                f"M {cx - rr},{cy - R - rr * 0.2} "
                f"a {rr},{rr} 0 1,1 {2 * rr},0 "
                f"a {rr},{rr} 0 1,1 {-2 * rr},0"
            ), fill="none", stroke=stroke_color, stroke_width=3)
            loop_path['marker-end'] = marker.get_funciri()
            dwg.add(loop_path)

            lab = ", ".join(sorted(labels))
            if lab:
                add_label(lab, cx, cy - R - rr - 18, is_nondet=is_nondet_loop)

        # Обычные рёбра
        for (u, v), labels in grouped.items():
            lab = ", ".join(sorted(labels))
            has_back = (v, u) in grouped
            rad = 0.25 if has_back and u < v else (-0.25 if has_back else (0.2 if pos[v][0] < pos[u][0] else 0.0))

            # Проверяем, является ли это недетерминированным переходом
            is_nondet = any((u, v, label) in nondet_edges for label in labels)

            draw_edge(u, v, lab, rad=rad, is_nondet=is_nondet)

        # Заголовок
        dwg.add(dwg.text("Недетерминированная диаграмма состояний (NFA)",
                       insert=(width / 2, 40), text_anchor="middle",
                       font_size=22, font_weight="bold", fill="#1A237E"))

        # Легенда
        legend_y = height - 30
        dwg.add(dwg.text("Красные переходы = недетерминированные",
                       insert=(width / 2, legend_y), text_anchor="middle",
                       font_size=14, fill="#D32F2F"))

        dwg.save()
        return filename
