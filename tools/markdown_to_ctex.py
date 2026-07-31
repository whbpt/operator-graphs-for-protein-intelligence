#!/usr/bin/env python3
"""Convert the project Chinese review Markdown into a typeset XeLaTeX source."""

from __future__ import annotations

import re
import sys
from pathlib import Path


INLINE_MATH = {
    "sequence -> structure": r"\text{sequence}\to\text{structure}",
    "N": "N",
    "L": "L",
    "q": "q",
    "H_seq": r"\mathcal H_{\mathrm{seq}}",
    "H_pos": r"\mathcal H_{\mathrm{pos}}",
    "H_cat": r"\mathcal H_{\mathrm{cat}}",
    "f": "f",
    "P_f(x)": r"P_f(x)",
    "X_f={x^(1),...,x^(N_f)}": r"X_f=\{x^{(1)},\ldots,x^{(N_f)}\}",
    "P_train(x)=sum_f pi_f P_f(x)": r"P_{\mathrm{train}}(x)=\sum_f\pi_fP_f(x)",
    "pi_f": r"\pi_f",
    "i": "i",
    "p_i(a)": r"p_i(a)",
    "T_ij(a,b)": r"T_{ij}(a,b)",
    "G_ij": r"G_{ij}",
    "G_ij p_j=0": r"G_{ij}p_j=0",
    "p_i^T G_ij=0": r"p_i^{\mathsf T}G_{ij}=0",
    "p_i x p_j": r"p_i\otimes p_j",
    "T_ij": r"T_{ij}",
    "z_ij": r"z_{ij}",
    "T_ij(a,b)=D_theta(z_ij;a,b)": r"T_{ij}(a,b)=D_\theta(z_{ij};a,b)",
    "r=T_theta(B)": r"r=T_\theta(B)",
    "m_si": r"m_{si}",
    "P_theta(x_i | x_-i)": r"P_\theta(x_i\mid x_{-i})",
    "P(structure,function | sequence,evidence)": (
        r"P(\mathrm{structure},\mathrm{function}\mid"
        r"\mathrm{sequence},\mathrm{evidence})"
    ),
    "P_0(x)": r"P_0(x)",
    "P_0": r"P_0",
    "P_D(x)": r"P_D(x)",
    "D": "D",
    "mu": r"\mu",
    "L_D": r"L_D",
    "u_i": r"u_i",
    "R_(>=3)": r"R_{\ge 3}",
    "Gamma_ij": r"\Gamma_{ij}",
    "Gamma_ij(x_i,x_j) = Gamma_ji(x_j,x_i)": (
        r"\Gamma_{ij}(x_i,x_j)=\Gamma_{ji}(x_j,x_i)"
    ),
    "P_i = I - 1 p_i^T": r"P_i=I-\mathbf 1p_i^{\mathsf T}",
    "j": "j",
    "specification -> backbone -> sequence -> validation": (
        r"\mathrm{specification}\to\mathrm{backbone}\to"
        r"\mathrm{sequence}\to\mathrm{validation}"
    ),
    "x_k": r"x_k",
}


DISPLAY_LINES = {
    "`{当前家族样本，语言模型先验，模板，化学环境} -> {序列状态，pair 状态，坐标状态} -> {结构，功能} <-> 条件蛋白质设计`": r"""
\begin{equation}
\begin{aligned}
&\left\{\text{当前家族样本、语言模型先验、模板、化学环境}\right\}
\longrightarrow
\left\{\text{序列、pair、坐标状态}\right\}\\
&\hspace{25mm}\longrightarrow
\left\{\text{结构、功能}\right\}
\rightleftarrows
\text{条件蛋白质设计}.
\end{aligned}
\end{equation}""",
    "`H_seq x H_pos x H_cat`。": r"""
\begin{equation}
\mathcal H_{\mathrm{seq}}\otimes
\mathcal H_{\mathrm{pos}}\otimes
\mathcal H_{\mathrm{cat}}.
\end{equation}""",
    "`G_ij = P_i T_ij P_j^T`，其中 `P_i = I - 1 p_i^T`。": r"""
\begin{equation}
G_{ij}=P_iT_{ij}P_j^{\mathsf T},
\qquad
P_i=I-\mathbf 1p_i^{\mathsf T}.
\end{equation}""",
    "`m_ij = || D_i^(1/2) G_ij D_j^(1/2) ||_F`。": r"""
\begin{equation}
m_{ij}=\left\|D_i^{1/2}G_{ij}D_j^{1/2}\right\|_F.
\end{equation}""",
    "`L_D(x) = log [ P_D(x) / P_0(x) ]`。": r"""
\begin{equation}
L_D(x)=\log\frac{P_D(x)}{P_0(x)}.
\end{equation}""",
    "`L_D = c + sum_i u_i(x_i) + sum_(i<j) G_ij(x_i,x_j) + R_(>=3)(x)`。": r"""
\begin{equation}
L_D=c+\sum_i u_i(x_i)
+\sum_{i<j}G_{ij}(x_i,x_j)+R_{\ge 3}(x).
\end{equation}""",
    "`||L_D-E_mu L_D||^2 = sum_i ||u_i||^2 + sum_(i<j)||G_ij||^2 + ||R_(>=3)||^2`。": r"""
\begin{equation}
\left\|L_D-\mathbb E_\mu L_D\right\|^2
=\sum_i\|u_i\|^2
+\sum_{i<j}\|G_{ij}\|^2
+\|R_{\ge 3}\|^2.
\end{equation}""",
    "`E_incompat = sum_(i<j)||Gamma_ij-Gamma_ji^T||^2 + sum_(i,j,k)||Delta_k Gamma_ij||^2`。": r"""
\begin{equation}
\mathcal E_{\mathrm{incompat}}
=\sum_{i<j}\left\|\Gamma_{ij}-\Gamma_{ji}^{\mathsf T}\right\|^2
+\sum_{i,j,k}\left\|\Delta_k\Gamma_{ij}\right\|^2.
\end{equation}""",
    "`P(sequence, conformational ensemble, free energy, kinetics, function, environment | specification)`。": r"""
\begin{equation}
P\!\left(
x,\mathcal A,\Delta G,\kappa,f,e\mid c
\right),
\end{equation}
其中 $\mathcal A$ 表示构象系综，$\Delta G$ 表示相对自由能，$\kappa$ 表示动力学量，$f$、$e$ 和 $c$ 分别表示功能、环境与设计条件。""",
    "`检索与先验 -> 多状态全原子设计 -> 独立计算测试 -> 多重实验 -> 不确定性感知更新`。": r"""
\begin{equation}
\boxed{
\text{检索与先验}
\to\text{多状态全原子设计}
\to\text{独立计算测试}
\to\text{多重实验}
\to\text{不确定性感知更新}}
\end{equation}""",
}


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,left=23mm,right=23mm,top=24mm,bottom=22mm]{geometry}
\usepackage{fontspec,xeCJK,amsmath,amssymb,microtype,enumitem,fancyhdr,titlesec,xcolor}
\usepackage[hidelinks]{hyperref}
\setmainfont{Times New Roman}
\setCJKmainfont{Songti SC}
\setCJKsansfont{HarmonyOS Sans SC}
\linespread{1.32}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.38em}
\setlength{\emergencystretch}{2em}
\setlist{nosep,leftmargin=2.2em}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{gray}蛋白质模型如何组织生物信息}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\titleformat{\section}{\CJKfamily{sf}\Large\bfseries}{ }{0pt}{}
\titleformat{\subsection}{\CJKfamily{sf}\large\bfseries}{ }{0pt}{}
\titlespacing*{\section}{0pt}{1.35em}{0.55em}
\titlespacing*{\subsection}{0pt}{1.0em}{0.35em}
\begin{document}
\begin{titlepage}
\vspace*{25mm}
{\CJKfamily{sf}\color{gray}\large 蛋白质建模综述\par}
\vspace{16mm}
{\CJKfamily{sf}\bfseries\fontsize{24}{34}\selectfont
蛋白质模型如何组织生物信息：\\[5mm]
进化、结构、功能与设计的多轴框架\par}
\vspace{13mm}
{\CJKfamily{sf}\Large 多轴信息分解、模型比较与理论展望\par}
\vfill
\rule{28mm}{1.6pt}
\end{titlepage}
\setcounter{page}{1}
"""


def escape_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def inline_tex(text: str) -> str:
    out: list[str] = []
    pos = 0
    for match in re.finditer(r"`([^`]+)`", text):
        out.append(escape_text(text[pos : match.start()]))
        token = match.group(1)
        if token in INLINE_MATH:
            out.append(r"\(" + INLINE_MATH[token] + r"\)")
        else:
            out.append(r"\texttt{" + escape_text(token) + "}")
        pos = match.end()
    out.append(escape_text(text[pos:]))
    return "".join(out)


def convert(src: Path, dst: Path) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    output = [PREAMBLE]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("# "):
            continue
        if line in DISPLAY_LINES:
            output.append(DISPLAY_LINES[line])
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            command = "section" if re.match(r"^[一二三四五六七八九十]+、", heading) else "subsection"
            output.append(f"\\{command}*{{{escape_text(heading)}}}")
            output.append(f"\\addcontentsline{{toc}}{{{command}}}{{{escape_text(heading)}}}")
            continue
        output.append(inline_tex(line) + r"\par")
    output.append("\\end{document}\n")
    dst.write_text("\n\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: markdown_to_ctex.py INPUT.md OUTPUT.tex")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
