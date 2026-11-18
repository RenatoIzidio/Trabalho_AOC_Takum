import math
import unittest
from fractions import Fraction as F
from spmv_benchmark import (quantize_ofp8_e4m3,quantize_ofp8_e5m2,quantize_fp16,
    quantize_fp32,quantize_bfloat16,reference_quad,compute_relative_2norm_error,
    compute_spmv_exact_fraction,SparseMatrixCSR,scalar_takum,execute_spmv_rvv_takum)

class TestBenchmark(unittest.TestCase):
    def test_fp8_codes_and_boundaries(self):
        for fn,m,bias,last in [(quantize_ofp8_e4m3,3,7,126),(quantize_ofp8_e5m2,2,15,123)]:
            values=[]
            for code in range(last+1):
                exp,frac=divmod(code,2**m)
                x=(frac/2**m if exp==0 else 1+frac/2**m)*2**(max(1,exp)-bias)
                values.append(x)
                for sign in (-1,1):
                    self.assertEqual(fn(sign*x),sign*x)
            for i,(lo,hi) in enumerate(zip(values,values[1:])):
                mid=(lo+hi)/2
                for sign in (-1,1):
                    for x,expected in [(math.nextafter(mid,0),lo),(mid,hi if i%2 else lo),
                                       (math.nextafter(mid,math.inf),hi)]:
                        self.assertEqual(fn(sign*x),sign*expected)
            self.assertEqual(math.copysign(1,fn(-0.0)),-1)
            self.assertTrue(math.isnan(fn(math.nan)))
            self.assertEqual(fn(-math.inf,True),-values[-1])
        self.assertEqual(quantize_ofp8_e4m3(464),448)
        self.assertTrue(math.isnan(quantize_ofp8_e4m3(math.nextafter(464,math.inf))))
        self.assertTrue(math.isnan(quantize_ofp8_e4m3(math.inf)))
        self.assertEqual(quantize_ofp8_e5m2(61440),math.inf)

    def test_wider_formats(self):
        for fn,p,emin,emax in [(quantize_fp16,10,-14,15),(quantize_fp32,23,-126,127),(quantize_bfloat16,7,-126,127)]:
            step=2.**(emin-p)
            self.assertEqual(fn(step/2),0)
            self.assertEqual(fn(1.5*step),2*step)
            mid=1+2.**(-p-1)
            self.assertEqual(fn(mid),1)
            self.assertEqual(fn(math.nextafter(mid,math.inf)),1+2.**-p)
            maximum=(2-2.**-p)*2.**emax
            self.assertEqual(fn(maximum),maximum)
            self.assertEqual(fn(maximum+2.**(emax-p-1)),math.inf)
            self.assertTrue(math.isnan(fn(math.nan)))
            self.assertEqual(fn(-math.inf),-math.inf)

    def test_metrics(self):
        err=compute_relative_2norm_error
        self.assertEqual(err([3,4],[F(0),F(5)]),math.sqrt(.4))
        self.assertEqual(err([0],[F(0)]),0)
        for v in (1,math.nan,math.inf):
            self.assertEqual(err([v],[F(0)]),math.inf)
        self.assertGreater(err([1.],[F(1)+F(1,2**100)]),0)
        with self.assertRaises(ValueError):
            err([], [F(1)])

    def test_csr(self):
        m=SparseMatrixCSR(3,3,[0,2,2,4],[2,0,1,1],[2,3,-1,4])
        self.assertEqual(compute_spmv_exact_fraction(m,[1,2,3]),[9,0,6])
        with self.assertRaises(ValueError):
            SparseMatrixCSR(1,3,[0,1],[3],[2])
        for w in (8,16,32):
            self.assertEqual(scalar_takum(m,[1,2,3],w,True),execute_spmv_rvv_takum(m,[1,2,3],w)[0])

    def test_quad_cancellation(self):
        m=SparseMatrixCSR.from_dense([[1,2.**-113,-1]])
        self.assertEqual(reference_quad(m,[1]*3),[0])
        self.assertEqual(compute_spmv_exact_fraction(m,[1]*3),[F(1,2**113)])
