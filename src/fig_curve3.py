"""Улучшенный рисунок кривых обучения (Obs. / Upper / Balanced / Lower) для полной статьи.

Каждая панель — один тип предпочтений:
  * полоса между Lower и Upper (80 % сценариев), Upper/Balanced/Lower — подогнанные кривые tau(n)=a-b*n^-c;
  * кружки — наблюдаемые средние гребневой регрессии;
  * треугольники / ромбы — случайный лес и градиентный бустинг из основного эксперимента (n = 50, 200);
  * пунктир — стартовое правило (WSM, для «вето» — OWA_0.5), правая ось — выигрыш над ним;
  * вертикальная линия — точка безубыточности n* сбалансированной кривой (если есть).
Общая легенда — отдельной полосой снизу.
python fig_curve3.py curve.csv full.csv curve_wmin.csv outdir
(вторая рецензия: треугольники — обучаемый WMin_lambda по сетке n, ромбы — обучаемая модель Шоке)
"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

d = pd.read_csv(sys.argv[1]); full = pd.read_csv(sys.argv[2]); cw = pd.read_csv(sys.argv[3]); out = sys.argv[4]
F = ["func", "m", "rho", "marg", "wconc", "noise"]
f = lambda n, a, b, c: a - b * n ** (-c)
nn = np.exp(np.linspace(np.log(10), np.log(500), 160))
FUNCS = ["linear", "concave", "interaction", "veto"]
T = {"en": dict(t={"linear": "(a) Linear", "concave": "(b) Concave", "interaction": "(c) Interaction", "veto": "(d) Veto"},
                obs="Obs. (ridge)", up="Upper", bal="Balanced", lo="Lower", band="10–90% of scenarios",
                rf="Learned WMin (obs.)", gb="Learned Choquet", cold="Cold-start rule",
                x="Past decisions n", y="Kendall τ", y2="Gain over rule", be="n* ≈ {:.0f}", nobe="no break-even"),
     "ru": dict(t={"linear": "(a) Линейные", "concave": "(b) Вогнутые", "interaction": "(c) Взаимодействие", "veto": "(d) Вето"},
                obs="Набл. (гребневая)", up="Верхняя", bal="Сбалансированная", lo="Нижняя", band="10–90 % сценариев",
                rf="Обуч. WMin (набл.)", gb="Обуч. модель Шоке", cold="Стартовое правило",
                x="Прошлые решения n", y="τ Кендалла", y2="Выигрыш над правилом", be="n* ≈ {:.0f}", nobe="нет точки безубыточности")}
FS = 23
FONT = dict(family="Times New Roman, Liberation Serif, serif", size=FS, color="black")
LO, HI = 0.45, 1.0
BAND = "rgba(56,161,105,0.17)"
C_RULE, C_ML, C_ML2, C_GB, C_NS = "#2b6cb0", "#1f5f3f", "#38a169", "#805ad5", "#dd6b20"

def fits(func):
    s = d[d.func == func]
    cell = s[s.method == "Ridge"].groupby(F + ["n"]).kendall.mean().reset_index()
    obs = cell.groupby("n").kendall.mean()
    q10 = cell.groupby("n").kendall.quantile(0.1); q90 = cell.groupby("n").kendall.quantile(0.9)
    base = s[s.method == ("OWA" if func == "veto" else "WSM")].kendall.mean()
    fit = lambda q: curve_fit(f, q.index.values, q.values, p0=[0.95, 1, 0.6], maxfev=20000)[0]
    pb, pl, pu = fit(obs), fit(q10), fit(q90)
    a, b, c = pb
    nstar = (b / (a - base)) ** (1 / c) if a > base else None
    fm = full[full.func == func].groupby(["method", "ntrain"]).kendall.mean()
    return obs, pb, pl, pu, base, nstar, fm

def panel(func, L, path, legend=False):
    obs, pb, pl, pu, base, nstar, fm = fits(func)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # полоса 10–90 %
    fig.add_trace(go.Scatter(x=np.r_[nn, nn[::-1]], y=np.r_[f(nn, *pu), f(nn, *pl)[::-1]], fill="toself", mode="none",
                             fillcolor=BAND, line=dict(width=0), name=L["band"], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[10, 500], y=[base, base], mode="lines", name=L["cold"], line=dict(color=C_RULE, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=nn, y=f(nn, *pu), mode="lines", name=L["up"], line=dict(color=C_ML2, width=2.4, dash="dashdot")))
    fig.add_trace(go.Scatter(x=nn, y=f(nn, *pb), mode="lines", name=L["bal"], line=dict(color=C_ML, width=3.8)))
    fig.add_trace(go.Scatter(x=nn, y=f(nn, *pl), mode="lines", name=L["lo"], line=dict(color=C_ML2, width=2.4, dash="dash")))
    fig.add_trace(go.Scatter(x=obs.index[obs.index > 0], y=obs[obs.index > 0].values, mode="markers", name=L["obs"],
                             marker=dict(size=13, color="white", line=dict(color=C_ML, width=2.2))))
    wo = cw[cw.func == func].groupby(F + ["n"]).kendall.mean().groupby("n").mean()
    fig.add_trace(go.Scatter(x=wo.index, y=wo.values, mode="lines+markers", name=L["rf"], line=dict(color="#dd6b20", width=1.6, dash="dot"),
                             marker=dict(symbol="triangle-up", size=14, color="#f6ad55", line=dict(color="#9c4221", width=1.6))))
    fig.add_trace(go.Scatter(x=[50, 200], y=[fm[("Choquet-learned", 50)], fm[("Choquet-learned", 200)]], mode="markers", name=L["gb"],
                             marker=dict(symbol="diamond", size=15, color=C_GB, line=dict(color="#44337a", width=1.5))))
    fig.add_trace(go.Scatter(x=[10, 500], y=[LO - base, HI - base], mode="markers", marker=dict(opacity=0),
                             showlegend=False, hoverinfo="skip"), secondary_y=True)
    if nstar and 10 <= nstar <= 500:
        fig.add_vline(x=nstar, line=dict(color=C_NS, width=2, dash="longdash"))
        fig.add_trace(go.Scatter(x=[nstar], y=[base], mode="markers", showlegend=False, marker=dict(size=14, color=C_NS, line=dict(color="white", width=2))))
        fig.add_annotation(x=np.log10(nstar), y=LO + 0.02, text="<b>"+L["be"].format(nstar)+"</b>", showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=6, font=dict(size=FS - 1, color=C_NS), bgcolor="white")
    else:
        fig.add_annotation(x=0.95, y=0.08, xref="paper", yref="paper", text="<b>"+L["nobe"]+"</b>", showarrow=False,
                           xanchor="right", font=dict(size=FS - 1, color="#c53030"), bgcolor="white")
    fig.add_annotation(x=0.02, y=0.985, xref="paper", yref="paper", text="<b>" + L["t"][func] + "</b>", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=FS + 1), bgcolor="white")
    fig.update_yaxes(range=[LO, HI], dtick=0.1, title_text=L["y"], showgrid=True, gridcolor="#e6e6e6", secondary_y=False)
    fig.update_yaxes(range=[LO - base, HI - base], tickformat="+.1f", tickmode="linear", tick0=0, dtick=0.1,
                     title_text=L["y2"], secondary_y=True, showgrid=False, color=C_RULE)
    fig.update_xaxes(type="log", tickvals=[10, 20, 50, 100, 200, 500], title_text=L["x"], showgrid=True, gridcolor="#e6e6e6")
    fig.update_layout(template="simple_white", font=FONT, width=640, height=470, margin=dict(l=80, r=90, t=12, b=10),
                      showlegend=legend)
    if legend:
        fig.update_layout(width=1760, height=210, legend=dict(orientation="h", x=0.5, xanchor="center", y=0.5, yanchor="middle",
                                                              font=dict(size=FS), entrywidth=430, entrywidthmode="pixels"),
                          margin=dict(l=20, r=20, t=0, b=0))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        for tr in fig.data:
            tr.x = [None] if tr.x is not None else tr.x
            tr.y = [None] if tr.y is not None else tr.y
        fig.layout.shapes = []; fig.layout.annotations = []
    fig.write_image(path, scale=3)
    return nstar

for lang, L in T.items():
    paths = []
    for func in FUNCS:
        p = f"{out}/_c2_{lang}_{func}.png"; ns = panel(func, L, p); paths.append(p)
        print(lang, func, ns)
    lp = f"{out}/_c2_{lang}_legend.png"; panel("linear", L, lp, legend=True)
    ims = [Image.open(p) for p in paths]; w, h = ims[0].size; gap = 40
    leg = Image.open(lp)
    # обрезаем легенду по содержимому
    import PIL.ImageOps as O
    bbox = O.invert(leg.convert("L")).getbbox(); leg = leg.crop(bbox)
    W = 2 * w + gap
    if leg.width > W:
        leg = leg.resize((W, int(leg.height * W / leg.width)))
    grid = Image.new("RGB", (W, 2 * h + gap + leg.height + 50), "white")
    for k, im in enumerate(ims):
        grid.paste(im, ((k % 2) * (w + gap), (k // 2) * (h + gap)))
    grid.paste(leg, ((W - leg.width) // 2, 2 * h + gap + 30))
    grid.save(f"{out}/fig_curve2_{lang}.png")
