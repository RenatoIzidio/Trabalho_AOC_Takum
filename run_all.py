#!/usr/bin/env python3
"""
run_all.py - Script Mestre de Execucao e Validacao de Todas as Fases do Projeto Integrador AOC.

Executa:
  1. Fase 1: Validacao da Aritmetica Takum (Decodificacao S, D, R, C, F e operacoes basicas)
  2. Fase 2: Validacao do Simulador RVV Zvtakum (vtadd.vv, vtsub.vv, vtmul.vv, vtdot.vv, vncvt.t.t)
  3. Fase 3: Benchmark Experimental SpMV (y = A * x) comparando Takum vs IEEE 754 (FP8, FP16, FP32)
"""

import sys
import unittest
import time
import test_takum
import test_rvv_takum
import test_spmv_benchmark
from spmv_benchmark import run_full_spmv_benchmark, export


def run_test_suite(suite_module, name: str) -> bool:
    print(f"\n======================================================================")
    print(f" EXECUCAO: {name}")
    print(f"======================================================================")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(suite_module)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    start_time = time.time()
    print("======================================================================")
    print(" PROJETO INTEGRADOR AOC - UFMG (2026/02)")
    print(" Extensao Vetorial RISC-V com Aritmetica Takum (Zvtakum)")
    print(" Integrantes: Luiz Fernando Esteves Soares, Renato Pedro Izidio Lopes,")
    print("              Tulio Cesar de Lima Miranda")
    print("======================================================================")

    # 1. Fase 1
    fase1_ok = run_test_suite(test_takum, "Fase 1: Modulo Aritmetico Takum (T8/T16/T32)")
    if not fase1_ok:
        print("[ERRO] Falha nos testes da Fase 1.")
        sys.exit(1)

    # 2. Fase 2
    fase2_ok = run_test_suite(test_rvv_takum, "Fase 2: ISA Zvtakum no Simulador Vetorial RVV 1.0")
    if not fase2_ok:
        print("[ERRO] Falha nos testes da Fase 2.")
        sys.exit(1)

    # 3. Fase 3
    print(f"\n======================================================================")
    print(f" EXECUCAO: Fase 3: Benchmark Experimental SpMV e Analise Arquitetural")
    print(f"======================================================================")
    if not run_test_suite(test_spmv_benchmark, "Fase 3: formatos, CSR e metricas"):
        sys.exit(1)
    bench_data = run_full_spmv_benchmark()
    export(bench_data)

    elapsed = time.time() - start_time
    print(f"\n======================================================================")
    print(f" TODOS OS TESTES E BENCHMARKS FORAM CONCLUIDOS COM SUCESSO! ({elapsed:.2f}s)")
    print(f" Grafico interativo gerado: spmv_error_distribution.html")
    print(f" Resultados salvos em: spmv_benchmark_results.json")
    print(f"======================================================================\n")


if __name__ == "__main__":
    main()
