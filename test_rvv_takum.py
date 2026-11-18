#!/usr/bin/env python3
"""
test_rvv_takum.py - Suite de Testes Automatizados para a Extensao Zvtakum no RVV.

Valida:
  - 5.2.1 vtadd.vv (Adicao vetorial em T8, T16, T32)
  - 5.2.2 vtsub.vv (Subtracao vetorial em T8, T16, T32)
  - 5.2.3 vtmul.vv (Multiplicacao vetorial em T8, T16, T32)
  - 5.2.4 vtdot.vv (Produto escalar vetorial com acumulo ampliado)
  - 5.2.5 vncvt.t.t (Conversao com reducao de largura de elemento)
  - Mascaramento e predicacao (vm=0 / v0)
  - Configuracao de tamanho de elemento (SEW) e agrupamento (LMUL)
  - Execucao de kernel SpMV (Sparse Matrix-Vector Multiplication)
"""

import math
import unittest
from fractions import Fraction
import takum as tk
from rvv_takum import RVVState, SparseMatrixCSR, execute_spmv_rvv_takum


class TestZvtakumInstructions(unittest.TestCase):
    """Testes unitarios para as cinco instrucoes da extensao Zvtakum."""

    def setUp(self):
        self.rvv = RVVState(vlen=256)

    def test_vtadd_vv_t16(self):
        """Testa adicao vetorial vtadd.vv em T16."""
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        a_vals = [1.0, 2.5, -3.0, 0.0]
        b_vals = [2.0, -1.5, 3.0, 5.25]
        self.rvv.write_vector(1, [tk.from_float(x, 16) for x in a_vals])
        self.rvv.write_vector(2, [tk.from_float(x, 16) for x in b_vals])

        self.rvv.vtadd_vv(vd=3, vs2=1, vs1=2)
        results = [tk.to_float(b, 16) for b in self.rvv.read_vector(3)]

        expected = [3.0, 1.0, 0.0, 5.25]
        for r, exp in zip(results, expected):
            self.assertAlmostEqual(r, exp, places=3)

    def test_vtsub_vv_t16(self):
        """Testa subtracao vetorial vtsub.vv em T16."""
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        a_vals = [5.0, 10.0, 2.5, 0.0]
        b_vals = [3.0, 10.0, -2.5, 4.0]
        self.rvv.write_vector(1, [tk.from_float(x, 16) for x in a_vals])
        self.rvv.write_vector(2, [tk.from_float(x, 16) for x in b_vals])

        self.rvv.vtsub_vv(vd=3, vs2=1, vs1=2)  # vs2 - vs1
        results = [tk.to_float(b, 16) for b in self.rvv.read_vector(3)]

        expected = [2.0, 0.0, 5.0, -4.0]
        for r, exp in zip(results, expected):
            self.assertAlmostEqual(r, exp, places=3)

    def test_vtmul_vv_t16(self):
        """Testa multiplicacao vetorial vtmul.vv em T16."""
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        a_vals = [2.0, -3.0, 0.5, 0.0]
        b_vals = [4.0, 2.0, -8.0, 100.0]
        self.rvv.write_vector(1, [tk.from_float(x, 16) for x in a_vals])
        self.rvv.write_vector(2, [tk.from_float(x, 16) for x in b_vals])

        self.rvv.vtmul_vv(vd=3, vs2=1, vs1=2)
        results = [tk.to_float(b, 16) for b in self.rvv.read_vector(3)]

        expected = [8.0, -6.0, -4.0, 0.0]
        for r, exp in zip(results, expected):
            self.assertAlmostEqual(r, exp, places=3)

    def test_all_widths_t8_t16_t32(self):
        """Verifica vtadd, vtsub e vtmul em T8, T16 e T32."""
        for sew in (8, 16, 32):
            self.rvv.vsetvli(rd=1, rs1_val=3, sew=sew, lmul=1)
            a = [1.0, 2.0, -1.0]
            b = [1.0, 0.5, 2.0]
            self.rvv.write_vector(1, [tk.from_float(x, sew) for x in a])
            self.rvv.write_vector(2, [tk.from_float(x, sew) for x in b])

            # vtadd.vv
            self.rvv.vtadd_vv(vd=3, vs2=1, vs1=2)
            r_add = [tk.to_float(x, sew) for x in self.rvv.read_vector(3)]
            self.assertAlmostEqual(r_add[0], 2.0, places=2)
            self.assertAlmostEqual(r_add[1], 2.5, places=2)
            self.assertAlmostEqual(r_add[2], 1.0, places=2)

            # vtmul.vv
            self.rvv.vtmul_vv(vd=4, vs2=1, vs1=2)
            r_mul = [tk.to_float(x, sew) for x in self.rvv.read_vector(4)]
            self.assertAlmostEqual(r_mul[0], 1.0, places=2)
            self.assertAlmostEqual(r_mul[1], 1.0, places=2)
            self.assertAlmostEqual(r_mul[2], -2.0, places=2)

    def test_vtdot_vv_widened_accumulation(self):
        """
        Testa vtdot.vv: produto escalar com acumulo em precisao ampliada.
        T8 com acumulo em T16, e T16 com acumulo em T32.
        """
        # Teste T8 -> T16
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=8, lmul=1)
        a8 = [1.0, 2.0, 0.5, -1.0]
        b8 = [2.0, 3.0, 4.0, 2.0]
        # dot = 1*2 + 2*3 + 0.5*4 + (-1)*2 = 2 + 6 + 2 - 2 = 8.0
        self.rvv.write_vector(1, [tk.from_float(x, 8) for x in a8])
        self.rvv.write_vector(2, [tk.from_float(x, 8) for x in b8])

        self.rvv.vtdot_vv(vd=3, vs2=1, vs1=2, widened=True)
        dot_res_t16 = self.rvv.read_scalar_result(3, 16)
        val = tk.to_float(dot_res_t16, 16)
        self.assertAlmostEqual(val, 8.0, places=2)

        # Teste T16 -> T32
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        a16 = [1.25, 2.5, -0.75, 4.0]
        b16 = [4.0, 2.0, 2.0, 0.5]
        # dot = 1.25*4 + 2.5*2 + (-0.75)*2 + 4.0*0.5 = 5.0 + 5.0 - 1.5 + 2.0 = 10.5
        self.rvv.write_vector(1, [tk.from_float(x, 16) for x in a16])
        self.rvv.write_vector(2, [tk.from_float(x, 16) for x in b16])

        self.rvv.vtdot_vv(vd=4, vs2=1, vs1=2, widened=True)
        dot_res_t32 = self.rvv.read_scalar_result(4, 32)
        val32 = tk.to_float(dot_res_t32, 32)
        self.assertAlmostEqual(val32, 10.5, places=4)

    def test_vncvt_t_t_narrowing_conversion(self):
        """
        Testa vncvt.t.t: conversao com reducao de largura de elemento (T16 -> T8 e T32 -> T16).
        """
        # Teste T16 -> T8 (SEW_dst = 8, SEW_src = 16)
        self.rvv.vsetvli(rd=1, rs1_val=5, sew=8, lmul=1)
        src_vals = [0.0, 1.0, 2.0, -1.5, 3.25]
        src_bits = [tk.from_float(x, 16) for x in src_vals]
        self.rvv.write_vector(2, src_bits, sew=16)

        self.rvv.vncvt_t_t(vd=1, vs2=2)
        dst_bits = self.rvv.read_vector(1, length=5, sew=8)
        dst_floats = [tk.to_float(b, 8) for b in dst_bits]

        for s_f, d_f in zip(src_vals, dst_floats):
            # T8 possui resolucao limitada, erro deve ser baixo
            self.assertAlmostEqual(s_f, d_f, delta=0.25)

        # Teste T32 -> T16 (SEW_dst = 16, SEW_src = 32)
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        src32 = [1.125, -2.0625, 100.5, 0.00390625]
        self.rvv.write_vector(2, [tk.from_float(x, 32) for x in src32], sew=32)
        self.rvv.vncvt_t_t(vd=4, vs2=2)
        dst16 = [tk.to_float(b, 16) for b in self.rvv.read_vector(4, length=4, sew=16)]

        for s_f, d_f in zip(src32, dst16):
            self.assertAlmostEqual(s_f, d_f, delta=1e-3)

    def test_predication_and_masking(self):
        """Testa operacoes vetoriais com predicacao (vm=0, mascara em v0)."""
        self.rvv.vsetvli(rd=1, rs1_val=4, sew=16, lmul=1)
        # Mascara em v0: ativa elementos 0 e 2 (bits 0b0101 = 0x5)
        self.rvv.v[0][0] = 0b0101

        init_dst = [10.0, 10.0, 10.0, 10.0]
        self.rvv.write_vector(3, [tk.from_float(x, 16) for x in init_dst])

        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [1.0, 1.0, 1.0, 1.0]
        self.rvv.write_vector(1, [tk.from_float(x, 16) for x in a_vals])
        self.rvv.write_vector(2, [tk.from_float(x, 16) for x in b_vals])

        # Executa com vm=0 (apenas elementos 0 e 2 devem mudar)
        self.rvv.vtadd_vv(vd=3, vs2=1, vs1=2, vm=0)
        res = [tk.to_float(b, 16) for b in self.rvv.read_vector(3)]

        self.assertAlmostEqual(res[0], 2.0, places=3)   # 1 + 1 = 2
        self.assertAlmostEqual(res[1], 10.0, places=3)  # mascarado: permanece 10.0
        self.assertAlmostEqual(res[2], 4.0, places=3)   # 3 + 1 = 4
        self.assertAlmostEqual(res[3], 10.0, places=3)  # mascarado: permanece 10.0

    def test_nar_propagation_in_vector(self):
        """Testa se NaR se propaga pontualmente no vetor sem contaminar elementos validos."""
        self.rvv.vsetvli(rd=1, rs1_val=3, sew=16, lmul=1)
        nar = tk.nar_pattern(16)
        v1 = [tk.from_float(1.0, 16), nar, tk.from_float(2.0, 16)]
        v2 = [tk.from_float(2.0, 16), tk.from_float(5.0, 16), tk.from_float(3.0, 16)]

        self.rvv.write_vector(1, v1)
        self.rvv.write_vector(2, v2)
        self.rvv.vtadd_vv(vd=3, vs2=1, vs1=2)

        res = self.rvv.read_vector(3)
        self.assertAlmostEqual(tk.to_float(res[0], 16), 3.0, places=3)
        self.assertEqual(res[1], nar)
        self.assertTrue(math.isnan(tk.to_float(res[1], 16)))
        self.assertAlmostEqual(tk.to_float(res[2], 16), 5.0, places=3)


class TestSparseMatrixVectorKernel(unittest.TestCase):
    """Testes para execucao do kernel SpMV no simulador RVV Takum."""

    def test_spmv_simple_csr(self):
        """Testa multiplicacao de matriz esparsa por vetor em T16."""
        dense_matrix = [
            [2.0, 0.0, 1.5],
            [0.0, 3.0, 0.0],
            [1.0, 0.5, 2.0]
        ]
        x = [1.0, 2.0, 2.0]
        # Esperado:
        # y[0] = 2*1 + 0*2 + 1.5*2 = 2 + 3 = 5.0
        # y[1] = 0*1 + 3*2 + 0*2 = 6.0
        # y[2] = 1*1 + 0.5*2 + 2*2 = 1 + 1 + 4 = 6.0
        csr = SparseMatrixCSR.from_dense(dense_matrix)
        y_sim, perf = execute_spmv_rvv_takum(csr, x, sew=16, vlen=128)

        self.assertEqual(len(y_sim), 3)
        self.assertAlmostEqual(y_sim[0], 5.0, places=2)
        self.assertAlmostEqual(y_sim[1], 6.0, places=2)
        self.assertAlmostEqual(y_sim[2], 6.0, places=2)
        self.assertGreater(perf.instructions_retired, 0)
        self.assertGreater(perf.vector_instructions, 0)
        self.assertGreater(perf.vector_element_ops, 0)

    def test_spmv_t8_and_t32(self):
        """Valida que SpMV opera corretamente tambem em T8 e T32."""
        dense_matrix = [
            [1.0, 0.0],
            [2.0, 1.0]
        ]
        x = [2.0, 3.0]
        # y[0] = 2.0, y[1] = 2*2 + 1*3 = 7.0
        csr = SparseMatrixCSR.from_dense(dense_matrix)

        y_t8, _ = execute_spmv_rvv_takum(csr, x, sew=8)
        self.assertAlmostEqual(y_t8[0], 2.0, delta=0.2)
        self.assertAlmostEqual(y_t8[1], 7.0, delta=0.5)

        y_t32, _ = execute_spmv_rvv_takum(csr, x, sew=32)
        self.assertAlmostEqual(y_t32[0], 2.0, places=4)
        self.assertAlmostEqual(y_t32[1], 7.0, places=4)



class TestFase2(unittest.TestCase):
    def test_grupos_fisicos(self):
        r = RVVState(64)
        for m in (1, 2, 4, 8):
            for w in (8, 16, 32):
                n = m * 64 // w
                self.assertEqual(r.vsetvli(0, 3*n, w, m), n)
                base = 32-m
                r.write_vector(base, list(range(n)))
                self.assertEqual(r.read_vector(base), list(range(n)))
                self.assertEqual(r.v[31][-w//8], n-1)
                with self.assertRaises(IndexError):
                    r.set_element(base, n, 0)
                if m > 1:
                    with self.assertRaises(ValueError):
                        r.vtmv_v_t(base+1, 1)

    def test_limites(self):
        r = RVVState(64)
        for reg in (-1, 32):
            with self.assertRaises(ValueError):
                r.vle(reg, 0)
        with self.assertRaises(ValueError):
            r.vsetvli(0, -1, 8)
        r.vsetvli(0, 1, 8, 8)
        with self.assertRaises(ValueError):
            r.vncvt_t_t(8, 16)

    def test_gather_bytes_e_mascara(self):
        r = RVVState(64)
        r.vsetvli(0, 5, 16, 2)
        r.mem_write_bytes(100, bytes(range(24)))
        r.write_vector(4, [3, 1, 9, 0, 5], 32)
        r.write_vector(2, [77]*8)
        r.v[0][0] = 0b10101
        r.vluxei(2, 100, 4, 32, vm=0)
        self.assertEqual(r.read_vector(2,8), [1027,77,2569,77,1541,77,77,77])
        self.assertEqual(r.perf.memory_load_bytes, 6)
        with self.assertRaises(ValueError):
            r.vle(0, 0, vm=0)

    def test_dot_arredondamento(self):
        for w in (8,16,32):
            r = RVVState(128)
            r.vsetvli(0,3,w)
            big = tk.max_pos(w)
            one = tk.from_float(1,w)
            r.write_vector(1,[big,one,tk.neg(big,w)])
            r.write_vector(2,[one]*3)
            bits = r.vtdot_vv(3,1,2)
            self.assertEqual(tk.to_float(bits,min(32,2*w)), int(w==32))
            r.vsetvli(0,0,w)
            self.assertEqual(r.vtdot_vv(3,1,2),bits)

    def test_reducao_mascarada(self):
        r = RVVState(64)
        r.vsetvli(0,3,16)
        one = tk.from_float(1,16)
        r.write_vector(1,[one,tk.nar_pattern(16),one])
        r.write_vector(2,[one]*3)
        r.v[0][0] = 5
        self.assertEqual(r.vtdot_vv(3,1,2,vm=0),tk.from_float(2,32))
        self.assertEqual(r.vtdot_vv(3,1,2),tk.nar_pattern(32))

    def test_narrow_empate(self):
        for w in (8,16):
            r = RVVState(128)
            r.vsetvli(0,5,w)
            low = 1 << (w-2)
            tie = (2*low+1) << (w-1)
            r.write_vector(2,[tie-1,tie,tie+1,tk.nar_pattern(2*w),0],2*w)
            r.vncvt_t_t(2,2)
            self.assertEqual(r.read_vector(2),[low,low,low+1,tk.nar_pattern(w),0])

    def test_spmv_nar_multibloco(self):
        for w in (8,16,32):
            a = SparseMatrixCSR.from_dense([[1]*17,[0]*17,[0]*16+[1]])
            y,_ = execute_spmv_rvv_takum(a,[float('nan')]+[1]*16,w,64)
            self.assertTrue(math.isnan(y[0]))
            self.assertEqual(y[1:],[0,1])

    def test_estimativa_lanes(self):
        for lanes in (1,2,4):
            r = RVVState(128,lanes)
            r.vsetvli(0,9,8)
            before = r.perf.cycles_estimated
            r.vtadd_vv(3,1,2)
            batches = math.ceil(9/lanes)
            self.assertEqual(r.perf.cycles_estimated-before,batches)
            before = r.perf.cycles_estimated
            r.vtdot_vv(3,1,2)
            self.assertEqual(r.perf.cycles_estimated-before,
                             batches+sum(math.ceil(k/lanes) for k in (4,2,1,1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
