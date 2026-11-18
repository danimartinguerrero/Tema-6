#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de datos para las figuras del Tema 6 (Teoría del potencial, parte I).

Produce, en ``figuras/datos/``, ficheros TikZ con trazados *vectoriales*
(líneas de campo y equipotenciales) calculados numéricamente, y en
``figuras/`` los mapas de color (PNG sin texto) de la figura de superposición.
Las figuras propiamente dichas (ejes, rótulos, conductores, cargas) se
construyen en TikZ/pgfplots dentro de ``figuras/fig-*.tex``, que hacen
``\\input`` de estos datos.

Física empleada
---------------
* Todas las configuraciones son axisimétricas con cargas puntuales sobre el
  eje; las líneas de campo se trazan en un plano meridiano con la ley de
  Coulomb 3D, E = sum_i q_i R_i / |R_i|^3 (unidades 1/(4 pi eps0) = 1).
* Las líneas parten de cada carga con ángulos polares alpha_k tales que
  cos(alpha_k) está equiespaciado: cada línea transporta el mismo flujo, de
  modo que la densidad de líneas que llegan a un conductor es proporcional a
  la densidad de carga inducida sigma.
* Integración RK4 sobre el campo de direcciones normalizado, con parada
  exacta sobre la superficie conductora (plano z = 0 o esfera r = a).
* La figura de superposición resuelve la ecuación de Laplace 2D en
  diferencias finitas (esquema de 5 puntos, factorización LU dispersa).

Uso:  python3 figuras/generar_figuras.py
"""
import os
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "datos")
os.makedirs(DATOS, exist_ok=True)

# ---------------------------------------------------------------- utilidades


def campo(p, cargas):
    """Campo de Coulomb (3D, restringido al plano meridiano) en el punto p."""
    E = np.zeros(2)
    for q, cx, cy in cargas:
        R = p - np.array([cx, cy])
        r = np.hypot(*R)
        if r < 1e-9:
            return np.zeros(2)
        E += q * R / r**3
    return E


def potencial(X, Y, cargas):
    V = np.zeros_like(X)
    for q, cx, cy in cargas:
        V += q / np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + 1e-12)
    return V


def trazar(p0, cargas, parar, signo=1.0, h=0.01, lmax=40.0, caja=None):
    """Integra dr/ds = signo * E/|E| desde p0 hasta que ``parar`` devuelva
    un punto final (o None para seguir)."""
    pts = [np.array(p0, float)]
    p = np.array(p0, float)

    def f(x):
        e = campo(x, cargas)
        n = np.hypot(*e)
        return signo * e / n if n > 0 else np.zeros(2)

    s = 0.0
    while s < lmax:
        k1 = f(p)
        k2 = f(p + 0.5 * h * k1)
        k3 = f(p + 0.5 * h * k2)
        k4 = f(p + h * k3)
        pn = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        fin = parar(p, pn)
        if fin is not None:
            pts.append(np.array(fin))
            return np.array(pts), True
        if caja is not None:
            x0, x1, y0, y1 = caja
            if not (x0 <= pn[0] <= x1 and y0 <= pn[1] <= y1):
                pts.append(pn)
                return np.array(pts), False
        pts.append(pn)
        p = pn
        s += h
    return np.array(pts), False


def angulos_flujo(n):
    """Ángulos polares con cos(alpha) equiespaciado (flujo igual por línea)."""
    k = np.arange(1, n + 1)
    return np.arccos(1 - 2 * (k - 0.5) / n)


def simplificar(pts, paso=0.035):
    """Submuestrea una polilínea manteniendo extremos (paso en unidades)."""
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        acc += np.hypot(*(b - a))
        if acc >= paso:
            out.append(b)
            acc = 0.0
    if np.hypot(*(out[-1] - pts[-1])) > 1e-9:
        out.append(pts[-1])
    return np.array(out)


def longitud(pts):
    return float(np.sum(np.hypot(*(np.diff(pts, axis=0).T))))


def tikz_linea(pts, estilo, esc=1.0, flecha=None):
    pts = simplificar(pts) * esc
    coords = " ".join(f"({x:.3f},{y:.3f})" for x, y in pts)
    st = estilo
    if flecha is not None:
        L = longitud(pts)
        pos = min(max(flecha * L, 0.25), max(L - 0.15, 0.05))
        st += f", flecha={pos:.3f}cm"
    return f"\\draw[{st}] plot coordinates {{{coords}}};\n"


def escribir(nombre, texto):
    with open(os.path.join(DATOS, nombre), "w", encoding="utf-8") as fh:
        fh.write("% Generado automáticamente por generar_figuras.py; no editar.\n")
        fh.write(texto)
    print("  ->", nombre)


def contornos(X, Y, Z, niveles, mascara=None):
    """Curvas de nivel (lista de polilíneas) vía matplotlib (sin dibujar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Zm = np.ma.array(Z, mask=mascara) if mascara is not None else Z
    fig = plt.figure()
    cs = plt.contour(X, Y, Zm, levels=niveles)
    segs = cs.allsegs
    plt.close(fig)
    return segs


# ===================================================== 1. Plano conductor


def plano():
    d = 1.0
    esc = 1.45                       # cm por unidad de d
    caja = (-2.3, 2.3, 0.0, 2.75)
    real = [(1.0, 0.0, d)]
    todo = [(1.0, 0.0, d), (-1.0, 0.0, -d)]

    def parar_plano(p, pn):
        if pn[1] <= 0.0:
            t = p[1] / (p[1] - pn[1])
            return p + t * (pn - p)
        return None

    def parar_imagen(p, pn):
        if np.hypot(pn[0], pn[1] + d) < 0.06:
            return np.array([0.0, -d + 0.06 * np.sign(pn[1] + d + 1e-12)])
        return None

    N = 9
    # (a) hipótesis ingenua: líneas radiales de q aislada
    txt = ""
    for a in angulos_flujo(N):
        for s in (+1, -1):
            u = np.array([s * np.sin(a), np.cos(a)])
            # tramo dentro de la región V (z>0)
            if u[1] < 0:
                t0 = d / (-u[1])
            else:
                t0 = 10
            tmax = 3.2
            t1 = min(t0, tmax)
            p_a = np.array([0, d]) + 0.07 * u
            p_b = np.array([0, d]) + t1 * u
            seg = np.array([p_a, p_b])
            seg = recortar(seg, (-2.3, 2.3, -1.0, 2.75))
            if seg is not None:
                txt += tikz_linea(seg, "flinea", esc, flecha=0.55)
            if u[1] < 0:  # prolongación que atraviesa el conductor (imposible)
                p_c = np.array([0, d]) + min(t0 + 0.9 / (-u[1]), tmax) * u
                seg2 = recortar(np.array([p_b, p_c]), (-2.3, 2.3, -1.0, 2.75))
                if seg2 is not None:
                    txt += tikz_linea(seg2, "flinea, prohibida", esc)
    escribir("plano-ingenuo.tex", txt)

    # (b) líneas reales (q + imagen), paran en z = 0; marcas de carga inducida
    txt, marcas = "", []
    for a in angulos_flujo(N):
        for s in (+1, -1):
            u = np.array([s * np.sin(a), np.cos(a)])
            p0 = np.array([0, d]) + 0.07 * u
            pts, llega = trazar(p0, todo, parar_plano, caja=caja, lmax=30)
            txt += tikz_linea(pts, "flinea", esc, flecha=0.45)
            if llega and abs(pts[-1][0]) < 2.2:
                marcas.append(pts[-1][0])
    txt += "".join(f"\\node[menos] at ({x*esc:.3f},{-0.16*esc:.3f}) {{}};\n" for x in marcas)
    escribir("plano-real.tex", txt)

    # (c) problema equivalente: q y -q; líneas virtuales punteadas en z<0
    caja2 = (-2.3, 2.3, -2.4, 2.75)
    txt = ""
    for a in angulos_flujo(N):
        for s in (+1, -1):
            u = np.array([s * np.sin(a), np.cos(a)])
            p0 = np.array([0, d]) + 0.07 * u
            pts, llega = trazar(p0, todo, parar_plano, caja=caja, lmax=30)
            txt += tikz_linea(pts, "flinea", esc, flecha=0.45)
            if llega:
                pv, _ = trazar(pts[-1] + np.array([0, -1e-4]), todo, parar_imagen,
                               caja=caja2, lmax=30)
                txt += tikz_linea(pv, "flinea, virtual", esc)
    # equipotenciales en V (z > 0)
    xs = np.linspace(-2.3, 2.3, 401)
    zs = np.linspace(0.0, 2.75, 241)
    X, Z = np.meshgrid(xs, zs)
    V = potencial(X, Z, todo)
    for nivel in [0.12, 0.25, 0.45, 0.8, 1.5]:
        for seg in contornos(X, Z, V, [nivel])[0]:
            if len(seg) > 3:
                txt += tikz_linea(seg, "equipot", esc)
    escribir("plano-imagen.tex", txt)


def recortar(seg, caja):
    """Recorta un segmento recto a la caja (Liang-Barsky simplificado)."""
    x0, x1, y0, y1 = caja
    p, q = seg[0], seg[1]
    t0, t1 = 0.0, 1.0
    dx, dy = q - p
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return None
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return np.array([p + t0 * (q - p), p + t1 * (q - p)])


# ===================================================== 2. Esfera conductora


def esfera():
    a, d = 1.0, 2.2
    esc = 0.98
    qp, dp = -a / d, a**2 / d
    todo = [(1.0, d, 0.0), (qp, dp, 0.0)]
    caja = (-2.1, 4.3, -2.35, 2.35)

    def parar_esfera(p, pn):
        if np.hypot(*pn) <= a:
            # bisección hasta la superficie
            lo, hi = 0.0, 1.0
            for _ in range(40):
                m = 0.5 * (lo + hi)
                if np.hypot(*(p + m * (pn - p))) > a:
                    lo = m
                else:
                    hi = m
            return p + lo * (pn - p)
        return None

    def parar_imagen(p, pn):
        if np.hypot(pn[0] - dp, pn[1]) < 0.05:
            return pn
        return None

    N = 11
    txt_r, txt_v, marcas = "", "", []
    for al in angulos_flujo(N):
        for s in (+1, -1):
            # el eje polar apunta hacia la esfera (-x) para repartir bien
            u = np.array([-np.cos(al), s * np.sin(al)])
            p0 = np.array([d, 0]) + 0.07 * u
            pts, llega = trazar(p0, todo, parar_esfera, caja=caja, lmax=30)
            txt_r += tikz_linea(pts, "flinea", esc, flecha=0.5)
            if llega:
                th = np.arctan2(pts[-1][1], pts[-1][0])
                marcas.append(th)
                pv, _ = trazar(pts[-1] * 0.999, todo, parar_imagen, h=0.005, lmax=5)
                txt_v += tikz_linea(pv, "flinea, virtual", esc)
    txt_m = "".join(
        f"\\node[menos] at ({0.86*a*np.cos(t)*esc:.3f},{0.86*a*np.sin(t)*esc:.3f}) {{}};\n"
        for t in marcas)
    escribir("esfera-real.tex", txt_r)
    escribir("esfera-marcas.tex", txt_m)
    escribir("esfera-virtual.tex", txt_v)

    # equipotenciales fuera de la esfera
    xs = np.linspace(caja[0], caja[1], 451)
    ys = np.linspace(caja[2], caja[3], 331)
    X, Y = np.meshgrid(xs, ys)
    V = potencial(X, Y, todo)
    mask = np.hypot(X, Y) < a
    txt = ""
    for nivel in [0.08, 0.16, 0.3, 0.55, 1.0, 1.8]:
        for seg in contornos(X, Y, V, [nivel], mascara=mask)[0]:
            if len(seg) > 3:
                txt += tikz_linea(seg, "equipot", esc)
    escribir("esfera-equipot.tex", txt)

    # --- carga en el interior de la esfera (problema inverso)
    s_in = 0.45
    qi, di = -a / s_in, a**2 / s_in
    todo_in = [(1.0, s_in, 0.0), (qi, di, 0.0)]
    escI = 1.6

    def parar_sup_int(p, pn):
        if np.hypot(*pn) >= a:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                m = 0.5 * (lo + hi)
                if np.hypot(*(p + m * (pn - p))) < a:
                    lo = m
                else:
                    hi = m
            return p + lo * (pn - p)
        return None

    def parar_img_ext(p, pn):
        if np.hypot(pn[0] - di, pn[1]) < 0.06:
            return pn
        return None

    txt, txtv = "", ""
    cajaI = (-1.6, 2.9, -1.5, 1.5)
    for al in angulos_flujo(9):
        for s in (+1, -1):
            u = np.array([np.cos(al), s * np.sin(al)])
            p0 = np.array([s_in, 0]) + 0.06 * u
            pts, llega = trazar(p0, todo_in, parar_sup_int, h=0.006, lmax=10)
            txt += tikz_linea(pts, "flinea", escI, flecha=0.5)
            if llega:
                pv, _ = trazar(pts[-1] * 1.001, todo_in, parar_img_ext, h=0.008,
                               lmax=6, caja=cajaI)
                txtv += tikz_linea(pv, "flinea, virtual", escI)
    escribir("esfera-interior.tex", txt)
    escribir("esfera-interior-virtual.tex", txtv)


# ===================================================== 3. Cavidad y apantallamiento


def cavidad():
    a, b = 1.0, 1.5          # radios interior y exterior de la corteza
    s = 0.38                 # posición de la carga interior
    q = 1.0
    D, Qe = 3.3, 1.6         # carga exterior (a la izquierda)
    esc = 1.2
    # interior: q + imagen respecto de r = a
    inter = [(q, s, 0.0), (-q * a / s, a**2 / s, 0.0)]
    # exterior: Qe + imagen respecto de r = b + carga central (total +q fuera)
    q1 = -Qe * b / D
    exter = [(Qe, -D, 0.0), (q1, -b**2 / D, 0.0), (q - q1, 0.0, 0.0)]

    def superficie(R, dentro):
        def parar(p, pn):
            r = np.hypot(*pn)
            if (dentro and r >= R) or (not dentro and r <= R):
                lo, hi = 0.0, 1.0
                for _ in range(40):
                    m = 0.5 * (lo + hi)
                    rm = np.hypot(*(p + m * (pn - p)))
                    if (dentro and rm < R) or (not dentro and rm > R):
                        lo = m
                    else:
                        hi = m
                return p + lo * (pn - p)
            return None
        return parar

    caja = (-4.3, 3.4, -2.45, 2.45)
    txt = ""
    marcas_int = []
    for al in angulos_flujo(7):
        for sg in (+1, -1):
            u = np.array([np.cos(al), sg * np.sin(al)])
            p0 = np.array([s, 0]) + 0.06 * u
            pts, llega = trazar(p0, inter, superficie(a, True), h=0.005, lmax=6)
            txt += tikz_linea(pts, "flinea", esc, flecha=0.5)
            if llega:
                marcas_int.append(np.arctan2(pts[-1][1], pts[-1][0]))
    # líneas que salen de Qe
    marcas_neg = []
    for al in angulos_flujo(10):
        for sg in (+1, -1):
            u = np.array([np.cos(al), sg * np.sin(al)])
            p0 = np.array([-D, 0]) + 0.07 * u
            pts, llega = trazar(p0, exter, superficie(b, False), caja=caja, lmax=30)
            txt += tikz_linea(pts, "flinea", esc, flecha=0.4)
            if llega:
                marcas_neg.append(np.arctan2(pts[-1][1], pts[-1][0]))
    # líneas que nacen en la parte positiva de la superficie exterior
    th = np.linspace(0, np.pi, 4001)
    Er = []
    for t in th:
        p = (b + 1e-6) * np.array([np.cos(t), np.sin(t)])
        Er.append(campo(p, exter) @ (p / np.hypot(*p)))
    Er = np.array(Er)
    flujo = np.where(Er > 0, Er, 0) * np.sin(th)          # dPhi ~ sigma dS
    acum = np.concatenate([[0], np.cumsum(0.5 * (flujo[1:] + flujo[:-1]) * np.diff(th))])
    # normalización: flujo por línea igual que en Qe (2*10 líneas por Qe)
    flujo_total_pos = acum[-1] * 2 * np.pi * b**2 / (4 * np.pi)   # en unidades de carga
    n_pos = int(round(flujo_total_pos / Qe * 10))
    marcas_pos = []
    for k in range(n_pos):
        objetivo = (k + 0.5) / n_pos * acum[-1]
        t = np.interp(objetivo, acum, th)
        marcas_pos.append(t)
        for sg in (+1, -1):
            p0 = (b + 0.005) * np.array([np.cos(t), sg * np.sin(t)])
            pts, _ = trazar(p0, exter, lambda p, pn: None, caja=caja, lmax=12)
            txt += tikz_linea(pts, "flinea", esc, flecha=0.35)
    escribir("cavidad-lineas.tex", txt)
    m = ""
    for t in marcas_int:
        m += f"\\node[menos] at ({(a+0.11)*np.cos(t)*esc:.3f},{(a+0.11)*np.sin(t)*esc:.3f}) {{}};\n"
    for t in marcas_neg:
        m += f"\\node[menos] at ({(b-0.11)*np.cos(t)*esc:.3f},{(b-0.11)*np.sin(t)*esc:.3f}) {{}};\n"
    for t in marcas_pos:
        for sg in (+1, -1):
            m += f"\\node[mas] at ({(b-0.11)*np.cos(t)*esc:.3f},{sg*(b-0.11)*np.sin(t)*esc:.3f}) {{}};\n"
    escribir("cavidad-marcas.tex", m)


# ===================================================== 4. Portada


def portada():
    """Carga q a distancia d = 1 de un plano conductor a tierra (z = 0).
    Líneas reales (z > 0) con degradado de color según el ángulo de salida;
    en z < 0 su reflejo especular, tenue: las líneas virtuales que mueren en
    la carga imagen -q (por simetría del dipolo, son exactamente el espejo)."""
    d = 1.0
    todo = [(1.0, 0.0, d), (-1.0, 0.0, -d)]
    esc = 1.5
    caja = (-4.2, 4.2, 0.0, 2.55)

    def parar(p, pn):
        if pn[1] <= 0.0:
            t = p[1] / (p[1] - pn[1])
            return p + t * (pn - p)
        return None

    # azul (líneas normales al plano) -> celeste -> verde agua -> oro -> naranja
    paleta = np.array([[60, 110, 200], [90, 169, 230], [90, 200, 190],
                       [214, 178, 94], [232, 137, 74]], float)

    def color(t):
        x = t * (len(paleta) - 1)
        i = min(int(x), len(paleta) - 2)
        c = paleta[i] + (x - i) * (paleta[i + 1] - paleta[i])
        return c.astype(int)

    real, espejo = "", ""
    N = 24
    for j, al in enumerate(angulos_flujo(N)):     # al medido desde -u_z (hacia el plano)
        for sg in (+1, -1):
            u = np.array([sg * np.sin(al), -np.cos(al)])
            p0 = np.array([0, d]) + 0.04 * u
            pts, _ = trazar(p0, todo, parar, caja=caja, lmax=40)
            r, g, bb = color(j / (N - 1))
            c = f"{{rgb,255:red,{r};green,{g};blue,{bb}}}"
            real += tikz_linea(pts, f"line width=0.55pt, color={c}", esc)
            esp = pts * np.array([1.0, -1.0])
            espejo += tikz_linea(esp, f"line width=0.45pt, color={c}, opacity=0.22", esc)
    escribir("portada.tex", espejo + real)


# ===================================================== 5. Superposición (Laplace 2D)


def superposicion():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    # dominio de cálculo (marco a tierra lejos) y ventana visible [0,4]x[0,3]
    h = 0.02
    xs = np.arange(-1.5, 5.5 + h / 2, h)
    ys = np.arange(-1.5, 4.5 + h / 2, h)
    X, Y = np.meshgrid(xs, ys)
    nx, ny = len(xs), len(ys)

    # geometría de los conductores (misma que en fig-superposicion.tex)
    rect = (X >= 0.85) & (X <= 1.85) & (Y >= 2.0) & (Y <= 2.72)
    barra = (X >= 2.35) & (X <= 3.55) & (Y >= 2.18) & (Y <= 2.34)
    circ = np.hypot(X - 2.18, Y - 1.62) <= 0.24
    tri_v = np.array([[0.32, 0.62], [0.82, 0.18], [1.85, 1.42]])

    def en_triangulo(P, T):
        def lado(p, a, b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        d1 = lado(P, T[0], T[1]); d2 = lado(P, T[1], T[2]); d3 = lado(P, T[2], T[0])
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(neg & pos)
    tri = en_triangulo((X, Y), tri_v)
    conductores = [tri, rect, barra, circ]
    borde = np.zeros_like(X, bool)
    borde[0, :] = borde[-1, :] = borde[:, 0] = borde[:, -1] = True
    fijo = borde | tri | rect | barra | circ

    idx = -np.ones(X.shape, int)
    libres = np.argwhere(~fijo)
    idx[~fijo] = np.arange(len(libres))
    n = len(libres)
    filas, cols, vals = [], [], []
    vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for k, (i, j) in enumerate(libres):
        filas.append(k); cols.append(k); vals.append(-4.0)
        for di, dj in vecinos:
            ii, jj = i + di, j + dj
            if not fijo[ii, jj]:
                filas.append(k); cols.append(idx[ii, jj]); vals.append(1.0)
    A = sp.csc_matrix((vals, (filas, cols)), shape=(n, n))
    lu = spl.splu(A)

    cmap = LinearSegmentedColormap.from_list(
        "tema", ["#0E1014", "#13283A", "#1F4B6E", "#3C83BD", "#5AA9E6",
                 "#A9C9B0", "#D6B25E", "#F2E3B3"])
    vis = (xs >= -1e-9) & (xs <= 4 + 1e-9)
    visy = (ys >= -1e-9) & (ys <= 3 + 1e-9)
    for m, cond in enumerate(conductores):
        b = np.zeros(n)
        Vfix = np.zeros_like(X)
        Vfix[cond] = 1.0
        for k, (i, j) in enumerate(libres):
            for di, dj in vecinos:
                ii, jj = i + di, j + dj
                if fijo[ii, jj]:
                    b[k] -= Vfix[ii, jj]
        sol = lu.solve(b)
        V = Vfix.copy()
        V[~fijo] = sol
        Vv = V[np.ix_(visy, vis)]
        Xv, Yv = X[np.ix_(visy, vis)], Y[np.ix_(visy, vis)]
        # mapa de color (sin texto, sin ejes); conductores se superponen en TikZ
        Vimg = np.sqrt(np.clip(Vv, 0, 1))      # realce perceptual suave
        from PIL import Image
        rgba = (cmap(Vimg[::-1, :]) * 255).astype(np.uint8)   # fila 0 = arriba
        img = Image.fromarray(rgba[:, :, :3]).resize(
            (rgba.shape[1] * 6, rgba.shape[0] * 6), Image.BICUBIC)
        img.save(os.path.join(AQUI, f"superposicion-{m+1}.png"))
        txt = ""
        mask = fijo[np.ix_(visy, vis)]
        for nivel in [0.05, 0.1, 0.2, 0.35, 0.55, 0.8]:
            for seg in contornos(Xv, Yv, Vv, [nivel], mascara=mask)[0]:
                if len(seg) > 3:
                    txt += tikz_linea(seg, "equisup")
        escribir(f"superposicion-{m+1}.tex", txt)


if __name__ == "__main__":
    print("Plano conductor"); plano()
    print("Esfera conductora"); esfera()
    print("Cavidad"); cavidad()
    print("Portada"); portada()
    print("Superposición (Laplace 2D)"); superposicion()
