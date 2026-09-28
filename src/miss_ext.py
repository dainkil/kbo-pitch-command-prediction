"""E1 보충 — far50(+chase) 중립 주입의 ε 확장 탐색 (폴드에서만 정한다)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "miss_magnitude.py")).read().split('head("1. 신호별')[0])
head("far50 / far50+chase / far50+far30 — 중립 ε 확장")
for nm, sigs in {"far50": ["mr_far50"], "far50+chase": ["mr_far50", "mr_chase"], "far50+far30": ["mr_far50", "mr_far30"], "far50+outsq": ["mr_far50", "mr_outsq"]}.items():
    for eps in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0): show(f"{nm} 중립 ε={eps}", evaluate(sigs, eps, True))
