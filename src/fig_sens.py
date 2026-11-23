"""Линейчатая диаграмма чувствительности: изменение tau Кендалла при смене уровня фактора (EN/RU).
python fig_sens.py full.csv outdir
"""
import sys
import pandas as pd
import plotly.graph_objects as go

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
METH = [("WSM-entropy", "#3182ce", {"en": "Weighted sum", "ru": "Взвешенная сумма"}),
        ("FIS-entropy", "#dd6b20", {"en": "Fuzzy FIS-3", "ru": "Нечёткая FIS-3"}),
        ("Ridge", "#22543d", {"en": "Ridge regression", "ru": "Гребневая регрессия"}),
        ("RandomForest", "#68d391", {"en": "Random forest", "ru": "Случайный лес"})]
FACT = [("rho", 0.0, 0.6, {"en": "Correlated criteria (ρ: 0 → 0.6)", "ru": "Коррелированные критерии (ρ: 0 → 0,6)"}),
        ("marg", "uniform", "sparse", {"en": "Sparse criteria (uniform → sparse)", "ru": "Разреженные критерии"}),
        ("m", 3, 5, {"en": "More criteria (m: 3 → 5)", "ru": "Больше критериев (m: 3 → 5)"}),
        ("wconc", "balanced", "concentrated", {"en": "Unequal true weights", "ru": "Неравные истинные веса"}),
        ("noise", 0.0, 0.2, {"en": "Noisy past decisions (σ: 0 → 0.2)", "ru": "Шум в решениях (σ: 0 → 0,2)"}),
        ("ntrain", 50, 200, {"en": "More past decisions (n: 50 → 200)", "ru": "Больше решений (n: 50 → 200)"})]
X = {"en": "Change in Kendall τ", "ru": "Изменение τ Кендалла"}
G = {"en": ("Criteria data", "Decision log"), "ru": ("Данные критериев", "Журнал решений")}
rows = []
for f, a, b, lab in FACT:
    g = d[d.method.isin([m for m, _, _ in METH])].groupby(["method", f]).kendall.mean()
    for m, _, _ in METH:
        rows.append(dict(f=f, m=m, delta=g[(m, b)] - g[(m, a)]))
R = pd.DataFrame(rows); print(R.pivot(index="f", columns="m", values="delta").round(3))
for lang in ("en", "ru"):
    ylab = [lab[lang] for *_, lab in FACT][::-1]
    fig = go.Figure()
    for m, c, nm in METH[::-1]:
        v = [R[(R.f == f) & (R.m == m)].delta.iloc[0] for f, *_ in FACT][::-1]
        t = [("%+.2f" % x).replace("-", "−") for x in v]
        if lang == "ru": t = [s.replace(".", ",") for s in t]
        fig.add_bar(y=ylab, x=v, orientation="h", name=nm[lang], marker=dict(color=c, line=dict(color="white", width=0.5)),
                    text=t, textposition="outside", textfont=dict(size=13), cliponaxis=False)
    fig.add_vline(x=0, line=dict(color="#333", width=1.5))
    # подложка для группы «журнал решений»
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-0.5, y1=1.5, fillcolor="rgba(15,118,110,0.07)", line_width=0, layer="below")
    fig.add_annotation(x=-0.195, y=5.3, text="<i>%s</i>" % G[lang][0], showarrow=False, xanchor="left", font=dict(size=14, color="#475569"))
    fig.add_annotation(x=-0.195, y=1.3, text="<i>%s</i>" % G[lang][1], showarrow=False, xanchor="left", font=dict(size=14, color="#0f766e"))
    fig.update_layout(template="simple_white", barmode="group", bargap=0.22, bargroupgap=0.05, width=820, height=545,
                      font=dict(family="Times New Roman, Liberation Serif, serif", size=16, color="black"),
                      margin=dict(l=10, r=20, t=10, b=10),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.19, traceorder="reversed", font=dict(size=15)))
    fig.update_xaxes(title_text=X[lang], range=[-0.2, 0.22], dtick=0.05, tickformat="+.2f", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(ticks="", showline=False)
    if lang == "ru": fig.update_layout(separators=", ")
    fig.write_image(f"{out}/fig_sens_{lang}.png", scale=3)
