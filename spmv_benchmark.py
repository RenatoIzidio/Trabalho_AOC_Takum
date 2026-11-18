"""SpMV sobre matrizes sinteticas. Metodologia e reproducao em FASE_03.md."""
import csv
import hashlib
import json
import math
import struct
import statistics
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
import takum as tk
from rvv_takum import SparseMatrixCSR, execute_spmv_rvv_takum, analyze_csr_complexity

def _ieee(x, fmt):
    try:
        return struct.unpack(fmt, struct.pack(fmt, x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)

def quantize_fp16(x):
    return _ieee(x, '>e')

def quantize_fp32(x):
    return _ieee(x, '>f')

def _binary(x, mantissa, emin, largest, overflow, saturate=False):
    if math.isnan(x) or x == 0:
        return x
    value = abs(x)
    if value <= largest * 2:
        exponent = max(emin, math.frexp(value)[1] - 1)
        step = 2.0 ** (exponent - mantissa)
        value = round(value / step) * step
    if value > largest:
        value = largest if saturate else overflow
    return math.copysign(value, x)

def quantize_ofp8_e4m3(x, saturate=False):
    return _binary(x, 3, -6, 448., math.nan, saturate)

def quantize_ofp8_e5m2(x, saturate=False):
    return _binary(x, 2, -14, 57344., math.inf, saturate)

def quantize_bfloat16(x):
    return _binary(x, 7, -126, float.fromhex('0x1.fep127'), math.inf)

def quantize_takum(x, n):
    return tk.to_float(tk.from_float(x,n),n)

def generate_benchmark_sparse_matrices(seed: int = 42) -> list[dict[str, Any]]:
    """
    Gera um conjunto diversificado de matrizes esparsas reproduzindo
    aplicacoes reais (CFD, redes eletricas, aprendizado de maquina, EDPs).
    """
    rng = np.random.default_rng(seed)
    matrices = []

    # 1. Matriz Laplaciana / Poisson 2D (5-pontos stencil) - 64x64
    n_grid = 8
    n_nodes = n_grid * n_grid
    lap_dense = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        lap_dense[i, i] = 4.0
        r, c = divmod(i, n_grid)
        if r > 0: lap_dense[i, i - n_grid] = -1.0
        if r < n_grid - 1: lap_dense[i, i + n_grid] = -1.0
        if c > 0: lap_dense[i, i - 1] = -1.0
        if c < n_grid - 1: lap_dense[i, i + 1] = -1.0
    matrices.append({
        "name": "poisson_2d_64x64",
        "category": "PDE / Finite Difference",
        "csr": SparseMatrixCSR.from_dense(lap_dense.tolist()),
        "x": rng.uniform(0.1, 2.0, size=n_nodes).tolist()
    })

    # 2. Matriz de Rede Eletrica / Power Grid (Banded com elementos diagonais fortes)
    n_power = 100
    pow_dense = np.zeros((n_power, n_power))
    for i in range(n_power):
        pow_dense[i, i] = rng.uniform(10.0, 50.0)
        for offset in (-3, -1, 1, 3):
            j = i + offset
            if 0 <= j < n_power and rng.random() > 0.3:
                pow_dense[i, j] = rng.uniform(-5.0, -0.5)
    matrices.append({
        "name": "power_grid_100x100",
        "category": "Power Systems / Structural",
        "csr": SparseMatrixCSR.from_dense(pow_dense.tolist()),
        "x": rng.uniform(-1.0, 1.0, size=n_power).tolist()
    })

    # 3. Matriz de Rede Neural Esparsa (ML Weights - 90% esparsidade, distribuicao Normal)
    n_ml_in, n_ml_out = 128, 64
    ml_dense = np.zeros((n_ml_out, n_ml_in))
    for r in range(n_ml_out):
        for c in range(n_ml_in):
            if rng.random() < 0.12:  # ~88% esparso
                ml_dense[r, c] = rng.normal(0.0, 1.5)
    matrices.append({
        "name": "sparse_mlp_weights_64x128",
        "category": "Machine Learning / Deep Learning",
        "csr": SparseMatrixCSR.from_dense(ml_dense.tolist()),
        "x": rng.uniform(0.0, 1.0, size=n_ml_in).tolist()
    })

    # 4. Matriz com Alto Alcance Dinamico (CFD / Convection-Diffusion ill-conditioned)
    # Testa os limites de subnormal e overflow dos formatos
    n_cfd = 80
    cfd_dense = np.zeros((n_cfd, n_cfd))
    for i in range(n_cfd):
        cfd_dense[i, i] = 10.0 ** rng.uniform(-2, 3)
        for _ in range(4):
            j = rng.integers(0, n_cfd)
            if j != i:
                cfd_dense[i, j] = (10.0 ** rng.uniform(-3, 2)) * rng.choice([-1.0, 1.0])
    matrices.append({
        "name": "cfd_multiscale_80x80",
        "category": "Computational Fluid Dynamics",
        "csr": SparseMatrixCSR.from_dense(cfd_dense.tolist()),
        "x": [10.0 ** rng.uniform(-1, 2) for _ in range(n_cfd)]
    })

    # 5. Colecao de Matrizes Esparsas Sinteticas Adicionais para Distribuicao Estatistica (N=30)
    for k in range(30):
        size = rng.integers(40, 80)
        density = rng.uniform(0.05, 0.15)
        mat = np.zeros((size, size))
        for r in range(size):
            mat[r, r] = rng.uniform(1.0, 10.0)
            n_nonzero = max(1, int(size * density))
            cols = rng.choice(size, size=n_nonzero, replace=False)
            for c in cols:
                scale = 10.0 ** rng.uniform(-2, 2)
                mat[r, c] = rng.uniform(-2.0, 2.0) * scale
        matrices.append({
            "name": f"synthetic_sparse_{k+1:02d}_{size}x{size}",
            "category": "Synthetic random sparse",
            "csr": SparseMatrixCSR.from_dense(mat.tolist()),
            "x": rng.uniform(-2.0, 2.0, size=size).tolist()
        })

    return matrices


# ==============================================================================
# Execucao do Benchmark e Calculo de Erro Relativo
# ==============================================================================

def compute_spmv_exact_fraction(matrix: SparseMatrixCSR, x: list[float]) -> list[Fraction]:
    """Calcula y_ref = A * x com aritmetica racional exata sobre as entradas binary64."""
    y_exact = []
    if len(x) != matrix.ncols:
        raise ValueError("dimensoes incompativeis")
    x_frac = [Fraction(v) for v in x]
    for r in range(matrix.nrows):
        start = matrix.row_ptr[r]
        end = matrix.row_ptr[r + 1]
        s = Fraction(0)
        for k in range(start, end):
            s += Fraction(matrix.values[k]) * x_frac[matrix.col_ind[k]]
        y_exact.append(s)
    return y_exact


def compute_spmv_in_format(
    matrix: SparseMatrixCSR,
    x: list[float],
    quantize_fn: Callable[[float], float]
) -> tuple[list[float], bool]:
    """
    Executa SpMV em um formato numerico especifico.
    Retorna o vetor y e uma flag booleana indicando se ocorreu estouro de alcance (overflow / inf / NaN).
    """
    y = []
    has_overflow = False
    if len(x) != matrix.ncols:
        raise ValueError("dimensoes incompativeis")
    x_q = [quantize_fn(v) for v in x]

    for r in range(matrix.nrows):
        start = matrix.row_ptr[r]
        end = matrix.row_ptr[r + 1]
        s = 0.0
        for k in range(start, end):
            val_q = quantize_fn(matrix.values[k])
            prod = quantize_fn(val_q * x_q[matrix.col_ind[k]])
            s = quantize_fn(s + prod)

            if math.isnan(s) or math.isinf(s):
                has_overflow = True

        y.append(s)
    return y, has_overflow



def compute_relative_2norm_error(y_approx, y_ref, precision=80):
    if len(y_approx) != len(y_ref):
        raise ValueError("dimensoes incompativeis")
    if any(not math.isfinite(v) for v in y_approx):
        return math.inf
    den = sum(v*v for v in y_ref)
    num = sum((Fraction(a)-b)**2 for a,b in zip(y_approx,y_ref))
    if not den:
        return math.inf if num else 0.
    ratio = Fraction(num, den)
    with localcontext() as ctx:
        ctx.prec = precision
        return float((Decimal(ratio.numerator)/Decimal(ratio.denominator)).sqrt())

def scalar_takum(matrix, x, w, tree=False):
    """Modelo escalar independente. tree=True reproduz a politica da Fase 2."""
    operands = [tk.from_float(v,w) for v in x]
    output = []
    wide = min(2*w,32)
    def rounded(q):
        return q if w == 32 else tk.to_fraction(tk.from_float(q,wide),wide)
    for first,last in zip(matrix.row_ptr,matrix.row_ptr[1:]):
        total = Fraction(0)
        acc = 0
        invalid = False
        while first < last:
            remaining = last-first
            capacity = 256//w
            count = remaining if remaining <= capacity else min(capacity,(remaining+1)//2)
            nodes = []
            for k in range(first,first+count):
                a = tk.from_float(matrix.values[k],w)
                b = operands[matrix.col_ind[k]]
                if tree:
                    if tk.nar_pattern(w) in (a,b):
                        invalid = True
                        break
                    nodes.append(rounded(tk.to_fraction(a,w)*tk.to_fraction(b,w)))
                else:
                    acc = tk.add(acc,tk.mul(a,b,w),w)
            if invalid:
                break
            while len(nodes)>1:
                nodes = [rounded(nodes[i]+nodes[i+1]) if i+1<len(nodes) else nodes[i]
                         for i in range(0,len(nodes),2)]
            if nodes:
                total += tk.to_fraction(tk.from_float(nodes[0],wide),wide)
            first += count
        if tree:
            acc = tk.nar_pattern(w) if invalid else tk.from_float(total,w)
        output.append(tk.to_float(acc,w))
    return output

def reference_quad(matrix, x):
    """Binary128 por operacao no dominio normal dos dados do benchmark."""
    def rne(q):
        if not q:
            return q
        magnitude = abs(q)
        exp = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        if magnitude < Fraction(2)**exp:
            exp -= 1
        if not -16382 <= exp <= 16382:
            raise ValueError("referencia fora da faixa normal validada")
        unit = Fraction(2)**(exp-112)
        return round(q/unit)*unit
    vector = list(map(Fraction,x))
    result = []
    for first,last in zip(matrix.row_ptr,matrix.row_ptr[1:]):
        total = Fraction(0)
        for k in range(first,last):
            total = rne(total + rne(Fraction(matrix.values[k])*vector[matrix.col_ind[k]]))
        result.append(total)
    return result


def run_full_spmv_benchmark():
    cases = generate_benchmark_sparse_matrices(42)
    rows, matrices, counters = [], [], []
    def record(name, fmt, kernel, y, reference, quad):
        error = compute_relative_2norm_error(y,reference)
        if error != compute_relative_2norm_error(y,reference,50):
            raise ArithmeticError("norma instavel entre 50 e 80 digitos")
        quad_error = compute_relative_2norm_error(y,quad)
        status = ('NaR' if fmt.startswith('T') else 'NaN') if any(math.isnan(v) for v in y) else (
            'infinity' if any(math.isinf(v) for v in y) else 'zero_reference' if not math.isfinite(error) else 'ok')
        rows.append(dict(matrix=name,format=fmt,kernel=kernel,status=status,
                         error_binary128=quad_error if math.isfinite(quad_error) else None,
                         error=error if math.isfinite(error) else None))
    for case in cases:
        name,m,x = case['name'],case['csr'],case['x']
        ref = compute_spmv_exact_fraction(m,x)
        quad = reference_quad(m,x)
        matrices.append(dict(quad_different_rows=sum(a!=b for a,b in zip(ref,quad)),matrix=name,family=case['category'],rows=m.nrows,cols=m.ncols,
                             nnz=m.nnz,sparsity=1-m.nnz/(m.nrows*m.ncols)))
        for fmt,fn in [('E4M3',quantize_ofp8_e4m3),('E5M2',quantize_ofp8_e5m2),
                       ('FP16',quantize_fp16),('BF16',quantize_bfloat16),('FP32',quantize_fp32)]:
            y,_ = compute_spmv_in_format(m,x,fn)
            record(name,fmt,'scalar',y,ref,quad)
        for w in (8,16,32):
            seq = scalar_takum(m,x,w)
            actual,perf = execute_spmv_rvv_takum(m,x,w)
            expected = scalar_takum(m,x,w,tree=True)
            if len(expected)!=len(actual) or any(a!=b and not(math.isnan(a) and math.isnan(b))
                                                 for a,b in zip(expected,actual)):
                raise AssertionError(f"RVV divergiu: {name} T{w}")
            counters.append(dict(matrix=name,width=w,verified_rows=len(actual),
                                 differing_from_sequential=sum(a!=b for a,b in zip(seq,actual)),**asdict(perf)))
            record(name,f'T{w}','scalar',seq,ref,quad)
            record(name,f'T{w}','RVV',actual,ref,quad)
    summary = []
    for fmt,kernel in sorted({(r['format'],r['kernel']) for r in rows}):
        group = [r for r in rows if (r['format'],r['kernel'])==(fmt,kernel)]
        finite = [r['error'] for r in group if r['error'] is not None]
        summary.append(dict(format=fmt,kernel=kernel,total=len(group),finite=len(finite),
                            stable=sum(e<1 for e in finite),median=statistics.median(finite) if finite else None))
    data = dict(seed=42,numpy=np.__version__,matrices=matrices,results=rows,
                summary=summary,rvv_metrics=counters,csr_analysis=analyze_csr_complexity())
    data['run_id'] = hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    return data

def generate_cumulative_error_plots(data, output_html='spmv_error_distribution.html'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = data['run_id']
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,formats in zip(axes,[('T8','E4M3','E5M2'),('T16','FP16','BF16'),('T32','FP32')]):
        right = max(r['error'] for r in data['results'] if r['format'] in formats and r['error'] is not None) * 1.15
        for row in data['summary']:
            fmt,kernel = row['format'],row['kernel']
            if fmt not in formats:
                continue
            errors = sorted(r['error'] for r in data['results'] if r['format']==fmt and
                            r['kernel']==kernel and r['error'] is not None)
            if errors:
                ax.step([0]+errors+[right],[0]+[(i+1)/row['total'] for i in range(len(errors))]+[len(errors)/row['total']],
                        where='post',label=f'{fmt} {kernel}')
        ax.set_xscale('symlog',linthresh=1e-10)
        ax.set_xlabel('Erro relativo em norma 2')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Fração das 34 matrizes')
    fig.suptitle('Distribuição acumulada do erro de SpMV')
    fig.tight_layout()
    fig.savefig('spmv_ecdf.svg',metadata={'Description':run})
    plt.close(fig)
    table = ''.join(f"<tr><td>{s['format']}</td><td>{s['kernel']}</td><td>{s['stable']}/{s['total']}</td><td>{s['median']:.5g}</td></tr>" for s in data['summary'])
    svg = "<svg" + Path("spmv_ecdf.svg").read_text().split("<svg",1)[1]
    details = []
    for model in data['summary']:
        fmt, kernel = model['format'], model['kernel']
        lines = []
        for r in data['results']:
            if (r['format'], r['kernel']) == (fmt, kernel):
                value = format(r['error'], '.6g') if r['error'] is not None else r['status']
                lines.append(f"<tr><td>{r['matrix']}</td><td>{value}</td></tr>")
        details.append(f"<details><summary>{fmt} · {kernel}</summary><table><tr><th>Matriz</th><th>Erro relativo</th></tr>{''.join(lines)}</table></details>")
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>SpMV · Fase 3</title>
    <style>body{{max-width:1100px;margin:3em auto;padding:0 1em;font:16px system-ui;color:#172734}}
    svg{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{text-align:left;padding:.4em 1em;border-bottom:1px solid #ddd}}
    summary{{cursor:pointer;padding:.5em}} th{{background:#eef4f5}}</style>
    <h1>Erro numérico de SpMV</h1><p>34 matrizes sintéticas · semente 42 · execução {run}</p>
    {svg}<h2>Resumo por formato</h2><table><tr><th>Formato</th><th>Kernel</th><th>Erro &lt; 1</th>
    <th>Mediana entre resultados finitos</th></tr>{table}</table>
    <h2>Resultados por matriz</h2>{''.join(details)}</html>"""
    Path(output_html).write_text(page,encoding="utf8")


def export(data):
    Path('spmv_benchmark_results.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    for key in ('summary','matrices','results','rvv_metrics'):
        with open(f'spmv_{key}.csv','w',newline='',encoding='utf8') as out:
            writer = csv.DictWriter(out,fieldnames=['run_id',*data[key][0]])
            writer.writeheader()
            writer.writerows(dict(run_id=data['run_id'],**r) for r in data[key])
    generate_cumulative_error_plots(data)

if __name__ == '__main__':
    export(run_full_spmv_benchmark())
