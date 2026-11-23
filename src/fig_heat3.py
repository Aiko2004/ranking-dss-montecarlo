"""Цветная тепловая карта tau Кендалла: метод × форма полезности (EN/RU).
python fig_heat3.py all3.csv outdir (вторая рецензия: + обучаемый WMin; жирным — все значения, равные максимуму при показанной точности)
"""
import sys
import pandas as pd
import plotly.graph_objects as go

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
FAM = [("crisp", "#2b6cb0", ["WSM-entropy", "TOPSIS-entropy", "OWA-andness0.5"]),
       ("fuzzy", "#c05621", ["FIS-entropy", "FIS-andness0.5", "TSK-centre-unif-c", "TSK-centre-unif-n", "FIS-WangMendel", "TSK-learned"]),
       ("ml", "#276749", ["WMin-learned", "Ridge", "Choquet-learned", "MonoGBM", "GradBoost", "RandomForest"])]
LAB = {"en": ({"WSM-entropy": "WSM", "TOPSIS-entropy": "TOPSIS", "OWA-andness0.5": "WMin<sub>0.5</sub>",
               "FIS-entropy": "Mamdani", "FIS-andness0.5": "Mamdani (λ=0.5)", "TSK-centre-unif-c": "TSK",
               "TSK-centre-unif-n": "TSK (λ=0.5)", "FIS-WangMendel": "Wang–Mendel", "TSK-learned": "TSK, learned",
               "WMin-learned": "WMin, learned", "Ridge": "Ridge", "Choquet-learned": "Choquet, learned", "MonoGBM": "Monotone GBM",
               "RandomForest": "Random forest", "GradBoost": "Grad. boosting"},
              ["Linear", "Concave", "Interaction", "Veto"], "Kendall τ",
              {"crisp": "Crisp rules", "fuzzy": "Fuzzy", "ml": "Learned"}),
       "ru": ({"WSM-entropy": "Взвеш. сумма", "TOPSIS-entropy": "TOPSIS", "OWA-andness0.5": "WMin<sub>0.5</sub>",
               "FIS-entropy": "Мамдани", "FIS-andness0.5": "Мамдани (λ=0,5)", "TSK-centre-unif-c": "TSK",
               "TSK-centre-unif-n": "TSK (λ=0,5)", "FIS-WangMendel": "Ванг–Мендель", "TSK-learned": "TSK обуч.",
               "WMin-learned": "WMin обуч.", "Ridge": "Гребневая регр.", "Choquet-learned": "Шоке обуч.", "MonoGBM": "Монот. GBM",
               "RandomForest": "Случайный лес", "GradBoost": "Град. бустинг"},
              ["Линейная", "Вогнутая", "Взаимод.", "Вето"], "τ Кендалла",
              {"crisp": "Чёткие", "fuzzy": "Нечёткие", "ml": "Обучаемые"})}
COLS = ["linear", "concave", "interaction", "veto"]
order = [m for _, _, ms in FAM for m in ms]
color = {m: c for _, c, ms in FAM for m in ms}
p = d[d.method.isin(order)].pivot_table(index="method", columns="func", values="kendall").loc[order, COLS]
best = {c: set(p.index[p[c].round(2) == p[c].max().round(2)]) for c in COLS}
for lang, (lab, cols, cb, fam) in LAB.items():
    z = p.values[::-1]; rows = order[::-1]
    txt = [[("<b>%.2f</b>" if m in best[c] else "%.2f") % p.loc[m, c] for c in COLS] for m in rows]
    if lang == "ru": txt = [[t.replace(".", ",") for t in r] for r in txt]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=list(range(len(rows))), text=txt, texttemplate="%{text}",
                               colorscale=[[0, "#fff5eb"], [0.35, "#fdd49e"], [0.6, "#a1d99b"], [0.8, "#41ab5d"], [1, "#005a32"]],
                               zmin=0.68, zmax=0.94, xgap=3, ygap=3, textfont=dict(size=22), showscale=False,
                               colorbar=dict(title=dict(text=cb, side="right"), thickness=14, len=0.9, tickformat=".2f")))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(rows))),
                     ticktext=['<span style="color:%s">%s</span>' % (color[m], lab[m]) for m in rows], showline=False, ticks="")
    fig.update_xaxes(side="top", showline=False, ticks="", tickangle=0)
    # разделители семейств и подписи групп
    for y in (5.5, 11.5):
        fig.add_shape(type="line", xref="paper", x0=-0.66, x1=1.0, y0=y, y1=y, line=dict(color="#333", width=1.6))
    pos = {"ml": 2.5, "fuzzy": 8.5, "crisp": 13}
    for k, c, _ in FAM:
        fig.add_annotation(xref="paper", x=-0.62, y=pos[k], xanchor="center", text="<b>%s</b>" % fam[k], textangle=-90, showarrow=False,
                           font=dict(color=c, size=21))
    fig.update_layout(template="simple_white", width=780, height=680, margin=dict(l=300, r=8, t=50, b=6),
                      font=dict(family="Times New Roman, Liberation Serif, serif", size=22, color="black"))
    fig.write_image(f"{out}/fig_heatmap_func{'' if lang == 'en' else '_ru'}.png", scale=3)
print(p.round(3)); print(best)
