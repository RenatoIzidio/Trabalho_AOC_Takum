"""
Testes automaticos do simulador Takum (Fase 1).

Rodar:  python3 -m unittest -v test_takum

Estrategia de verificacao:
  * as operacoes sao conferidas contra resultados exatos de Fraction;
  * o criterio de arredondamento considera a fronteira de codificacao em
    n+1 bits, e empates sao resolvidos para o codigo par (ties to even);
  * exemplos de resultado esperado usam constantes independentes da
    implementacao, incluindo constantes publicadas pela libtakum.
    Nao ha dependencia de biblioteca externa na execucao dos testes.

IMPORTANTE: em T8 a distancia real entre valores vizinhos pode variar muito
quando ha ghost bits; os limiares sao em espaco de codificacao.
"""
import math
import random
import unittest
from fractions import Fraction

import takum as tk
from takum import T8, T16, T32


def midpoint(P: int, n: int) -> Fraction:
    """Valor do ponto medio (espaco da codificacao) entre os padroes P e P+1."""
    return tk.to_fraction((P << 1) | 1, n + 1)


def correct_rounding(x: Fraction, T: int, n: int) -> bool:
    nar = tk.nar_pattern(n)
    if x == 0:
        return T == 0
    if T == 0 or T == nar:
        return False                                  # nao-nulo nunca vira 0/NaR
    if (T >> (n - 1)) != (1 if x < 0 else 0):
        return False                                  # sinal errado
    if T not in (tk.min_pos(n), tk.max_neg(n)):       # tem vizinho inferior finito
        lower = midpoint(T - 1, n)
        if x < lower or (x == lower and (T & 1)):
            return False
    if T not in (tk.max_pos(n), tk.min_neg(n)):       # tem vizinho superior finito
        upper = midpoint(T, n)
        if x > upper or (x == upper and (T & 1)):
            return False
    return True


def exact(op: str, a: int, b: int, n: int):
    fa, fb = tk.to_fraction(a, n), tk.to_fraction(b, n)
    if fa is None or fb is None:
        return None
    return {"add": fa + fb, "sub": fa - fb, "mul": fa * fb}[op]


OPS = {"add": tk.add, "sub": tk.sub, "mul": tk.mul}


def special_patterns(n):
    return [0, tk.nar_pattern(n), tk.max_pos(n), tk.min_pos(n), tk.max_neg(n),
            tk.min_neg(n), tk.from_float(1, n), tk.from_float(-1, n),
            tk.from_float(2, n), tk.from_float(0.5, n)]


def random_pair(rng, n):
    mask = (1 << n) - 1
    mode = rng.randrange(5)
    a = rng.getrandbits(n)
    if mode == 0:
        b = rng.getrandbits(n)
    elif mode == 1:                                   # vizinhos: cancelamento em sub
        b = (a + rng.randint(-4, 4)) & mask
    elif mode == 2:                                   # quase opostos: cancelamento em add
        b = (tk.neg(a, n) + rng.randint(-4, 4)) & mask
    elif mode == 3:                                   # valores "comuns"
        a = tk.from_float(rng.uniform(-100, 100), n)
        b = tk.from_float(rng.uniform(-100, 100), n)
    else:                                             # patologicos x aleatorio
        a = rng.choice(special_patterns(n))
        b = rng.getrandbits(n) if rng.random() < .5 else rng.choice(special_patterns(n))
    return a, b


class TestFields(unittest.TestCase):
    def test_known_encodings_t16(self):
        table = {1.0: 0x4000, -1.0: 0xC000, 2.0: 0x4800, 0.5: 0x3800,
                 1.5: 0x4400, -1.5: 0xBC00, 32.0: 0x5400, 2.0 ** -5: 0x2C00}
        for x, bits in table.items():
            self.assertEqual(tk.from_float(x, T16), bits, hex(bits))
            self.assertEqual(tk.to_float(bits, T16), x)

    def test_known_encodings_t8_t32(self):
        for x, bits in {1.0: 0x40, -1.0: 0xC0, 2.0: 0x48, 0.5: 0x38}.items():
            self.assertEqual(tk.from_float(x, T8), bits)
        self.assertEqual(tk.from_float(1.0, T32), 0x40000000)
        self.assertEqual(tk.from_float(-1.0, T32), 0xC0000000)

    def test_reference_constants_libtakum(self):
        # Valores fixados em takum-arithmetic/libtakum/takum.h (takums lineares).
        # Independentes da nossa rotina de decodificacao/codificacao.
        known = {
            "one": (1.0, (64, 16384, 1073741824)),
            "pi": (math.pi, (74, 19016, 1246265045)),
            "ln2": (math.log(2), (59, 15127, 991371647)),
            "phi": ((1 + math.sqrt(5)) / 2, (69, 17650, 1156692942)),
        }
        for label, (value, patterns) in known.items():
            for n, expected in zip((T8, T16, T32), patterns):
                self.assertEqual(tk.from_float(value, n), expected,
                                 f"{label} em T{n}")

    def test_fields_positive_char(self):
        f = tk.decode_fields(0x5400, T16)              # 32 = 2^5
        self.assertEqual((f.S, f.D, f.R, f.r, f.C, f.c, f.F, f.p, f.e),
                         (0, 1, 0b010, 2, 0b10, 5, 0, 9, 5))

    def test_fields_negative_char(self):
        f = tk.decode_fields(0x2C00, T16)              # 2^-5
        self.assertEqual((f.S, f.D, f.R, f.r, f.C, f.c, f.e),
                         (0, 0, 0b101, 2, 0b10, -5, -5))

    def test_fields_negative_number(self):
        f = tk.decode_fields(0xBC00, T16)              # -1.5
        self.assertEqual((f.S, f.D, f.r, f.c, f.e, f.F, f.p), (1, 0, 0, -1, 0, 0x400, 11))
        self.assertEqual(tk.to_fraction(0xBC00, T16), Fraction(-3, 2))

    def test_t8_ghost_bits(self):
        f = tk.decode_fields(0x7F, T8)                 # maior positivo de T8
        self.assertEqual((f.r, f.C, f.c, f.p, f.c_bits_present, f.ghost_bits),
                         (7, 0b1110000, 239, 0, 3, 4))
        self.assertEqual(tk.to_fraction(0x7F, T8), Fraction(2) ** 239)
        self.assertEqual(tk.to_fraction(0x01, T8), Fraction(1, 2 ** 239))

    def test_specials(self):
        for n in (T8, T16, T32):
            self.assertEqual(tk.decode_fields(0, n).kind, "zero")
            self.assertEqual(tk.decode_fields(1 << (n - 1), n).kind, "nar")
            self.assertEqual(tk.to_fraction(0, n), 0)
            self.assertIsNone(tk.to_fraction(1 << (n - 1), n))


class TestFormatProperties(unittest.TestCase):
    def test_exhaustive_properties(self):
        # n pequenos: todos os padroes (inclui n < 12 com ghost bits)
        for n in (8, 10, 12, 16):
            nar = tk.nar_pattern(n)
            ordered = [(p - (1 << n) if p >= nar else p, p) for p in range(1 << n)]
            ordered.sort()                              # ordem de inteiro com sinal
            prev = None
            for _, p in ordered:
                if p == nar:
                    continue
                v = tk.to_fraction(p, n)
                if prev is not None:                    # monotonicidade estrita
                    self.assertLess(prev, v, (n, hex(p)))
                prev = v
                self.assertEqual(tk.to_fraction(tk.neg(p, n), n), -v)   # negacao
                self.assertEqual(tk.from_float(v, n) if v.denominator & (v.denominator - 1) == 0
                                 else None, p)                          # ida e volta

    def test_dynamic_range_symmetry(self):
        for n in (T16, T32):
            p = n - 12
            hi = (2 - Fraction(1, 2 ** p)) * Fraction(2) ** 254
            lo = (1 + Fraction(1, 2 ** p)) * Fraction(1, 2 ** 255)
            self.assertEqual(tk.to_fraction(tk.max_pos(n), n), hi)
            self.assertEqual(tk.to_fraction(tk.min_pos(n), n), lo)
            self.assertEqual(tk.to_fraction(tk.max_neg(n), n), -hi)
            self.assertEqual(tk.to_fraction(tk.min_neg(n), n), -lo)

    def test_roundtrip_t32_random(self):
        rng = random.Random(1)
        for _ in range(20000):
            p = rng.getrandbits(32)
            if p == tk.nar_pattern(32):
                continue
            v = tk.to_fraction(p, 32)
            self.assertEqual(tk.to_fraction(tk.neg(p, 32), 32), -v)
            if v != 0:
                kind, neg_, N, E = tk._unpack(p, 32)
                self.assertEqual(tk._round_pack(neg_, N, E, 32), p)


class TestSpecialCases(unittest.TestCase):
    def test_nar_propagation(self):
        for n in (T8, T16, T32):
            nar = tk.nar_pattern(n)
            for x in (0, 1, nar, tk.from_float(3, n)):
                for op in OPS.values():
                    self.assertEqual(op(x, nar, n), nar)
                    self.assertEqual(op(nar, x, n), nar)

    def test_zero_and_identity(self):
        rng = random.Random(2)
        for n in (T8, T16, T32):
            for _ in range(500):
                x = rng.getrandbits(n)
                if x == tk.nar_pattern(n):
                    continue
                self.assertEqual(tk.add(x, 0, n), x)
                self.assertEqual(tk.add(0, x, n), x)
                self.assertEqual(tk.mul(x, 0, n), 0)
                self.assertEqual(tk.sub(x, x, n), 0)
                self.assertEqual(tk.mul(x, tk.from_float(1, n), n), x)
                self.assertEqual(tk.mul(x, tk.from_float(-1, n), n), tk.neg(x, n))

    def test_saturation(self):
        for n in (T8, T16, T32):
            mp, mn, lp, ln = tk.max_pos(n), tk.max_neg(n), tk.min_pos(n), tk.min_neg(n)
            self.assertEqual(tk.mul(mp, mp, n), mp)             # estouro -> satura
            self.assertEqual(tk.mul(mp, mn, n), mn)
            self.assertEqual(tk.add(mp, mp, n), mp)
            self.assertEqual(tk.add(mn, mn, n), mn)
            self.assertEqual(tk.mul(lp, lp, n), lp)             # subfluxo -> menor != 0
            self.assertEqual(tk.mul(lp, ln, n), ln)
        self.assertEqual(tk.sub(0x0002, 0x0001, T16), 1)        # resultado minusculo != 0

    def test_float_conversion(self):
        self.assertEqual(tk.from_float(float("inf"), T16), tk.nar_pattern(T16))
        self.assertEqual(tk.from_float(float("nan"), T16), tk.nar_pattern(T16))
        self.assertEqual(tk.from_float(0.0, T16), 0)
        rng = random.Random(3)
        for n in (T8, T16, T32):
            for _ in range(3000):
                x = rng.uniform(-1e6, 1e6) * 2.0 ** rng.randint(-60, 60)
                T = tk.from_float(x, n)
                self.assertTrue(correct_rounding(Fraction(x), T, n), (n, x))

    def test_decimal_as_exact_rational(self):
        # 0.1 decimal nao deve passar antes por float64.
        for n in (T8, T16, T32):
            for s in ("0.1", "-0.1", "1.2", "3.141592653589793238462643383279",
                      "1e-70", "-2.5e40"):
                self.assertEqual(tk.from_float(s, n),
                                 tk.from_float(Fraction(s), n))
        self.assertEqual(tk._parse_operand("0.1", T16), tk.from_float("0.1", T16))
        self.assertEqual(tk._parse_operand("0x4000", T16), 0x4000)

    def test_invalid_inputs(self):
        for n in (-1, 0, 4):
            with self.assertRaises(ValueError):
                tk.from_float(0, n)
            with self.assertRaises(ValueError):
                tk.decode_fields(0, n)
            with self.assertRaises(ValueError):
                tk.nar_pattern(n)
        for n in (8.0, "8", None, True):
            with self.assertRaises(TypeError):
                tk.from_float(0, n)
        with self.assertRaises(ValueError):
            tk.decode_fields(256, T8)
        with self.assertRaises(ValueError):
            tk.decode_fields(-1, T8)
        with self.assertRaises(TypeError):
            tk.decode_fields(1.0, T8)
        with self.assertRaises(ValueError):
            tk._parse_operand("0x100", T8)
        with self.assertRaises(ValueError):
            tk._round_pack(False, 0, 0, T8)
        with self.assertRaises(TypeError):
            tk._round_pack(False, 1, 1.5, T8)
        with self.assertRaises(ValueError):
            tk.Takum.from_float(1, T8) + tk.Takum.from_float(1, T16)

    def test_rounding_ties_to_even(self):
        # Resultados conhecidos em T8: 1=0x40, 1.125=0x41 e 1.25=0x42.
        # Empates: 1.0625 -> 0x40 (par), 1.1875 -> 0x42 (par).
        examples = {
            Fraction(17, 16): 0x40,
            Fraction(19, 16): 0x42,
            Fraction(-17, 16): 0xC0,
            Fraction(-19, 16): 0xBE,
            Fraction(17, 16) - Fraction(1, 1 << 20): 0x40,
            Fraction(17, 16) + Fraction(1, 1 << 20): 0x41,
        }
        for x, expected in examples.items():
            self.assertEqual(tk.from_float(x, T8), expected, str(x))
            self.assertTrue(correct_rounding(x, expected, T8), str(x))
        self.assertFalse(correct_rounding(Fraction(17, 16), 0x41, T8))
        self.assertFalse(correct_rounding(Fraction(19, 16), 0x41, T8))

        for n in (T8, T16, T32):
            one = tk.from_float(1, n)
            v0, v1 = tk.to_fraction(one, n), tk.to_fraction(one + 1, n)
            m = (v0 + v1) / 2
            self.assertEqual(tk.from_float(m, n), one)  # codigo par
            self.assertEqual(tk.from_float(-m, n), tk.neg(one, n))
            self.assertEqual(tk.from_float(m + (v1-v0)/16, n), one + 1)

    def test_ghost_bits_rounding_t8(self):
        # Para T8, as fronteiras sao definidas pelos bits descartados de C;
        # a media ARITMETICA dos dois valores pode divergir do limiar Takum.
        low, high = tk.to_fraction(1, T8), tk.to_fraction(2, T8)
        threshold = midpoint(1, T8)
        self.assertNotEqual(threshold, (low + high) / 2)
        self.assertEqual(tk.from_float(threshold, T8), 2)
        self.assertEqual(tk.from_float(threshold - threshold / 2**30, T8), 1)
        self.assertEqual(tk.from_float(threshold + threshold / 2**30, T8), 2)

    def test_dyadic_and_rational_encoder_agree(self):
        # Duas rotas distintas: empacotamento de inteiros e racional exato.
        rng = random.Random(2026)
        for n in (T8, T16, T32):
            for _ in range(3000):
                x = Fraction(rng.randrange(-100000, 100000), 1 << rng.randrange(0, 90))
                self.assertEqual(tk.from_float(x, n),
                                 tk._from_rational_exact(x, n) if x else 0)

    def test_operator_class(self):
        a, b = tk.Takum.from_float(1.5, T16), tk.Takum.from_float(2.25, T16)
        self.assertEqual(float(a + b), 3.75)
        self.assertEqual(float(b - a), 0.75)
        self.assertEqual(float(a * b), 3.375)
        self.assertEqual(float(-a), -1.5)
        self.assertEqual(float(2 * a), 3.0)


class TestArithmetic(unittest.TestCase):
    def _check(self, op, a, b, n):
        x = exact(op, a, b, n)
        r = OPS[op](a, b, n)
        if x is None:
            self.assertEqual(r, tk.nar_pattern(n))
        else:
            self.assertTrue(correct_rounding(x, r, n),
                            f"{op} n={n} a=0x{a:X} b=0x{b:X} -> 0x{r:X}")

    def test_t8_exhaustive(self):
        for op in OPS:
            for a in range(256):
                for b in range(256):
                    self._check(op, a, b, T8)

    def test_commutativity_t8_exhaustive(self):
        for a in range(256):
            for b in range(256):
                self.assertEqual(tk.add(a, b, T8), tk.add(b, a, T8))
                self.assertEqual(tk.mul(a, b, T8), tk.mul(b, a, T8))

    def test_t16_random(self):
        rng = random.Random(16)
        for _ in range(15000):
            a, b = random_pair(rng, T16)
            for op in OPS:
                self._check(op, a, b, T16)

    def test_t32_random(self):
        rng = random.Random(32)
        for _ in range(15000):
            a, b = random_pair(rng, T32)
            for op in OPS:
                self._check(op, a, b, T32)

    def test_t16_special_cross_product(self):
        sp = special_patterns(T16)
        for a in sp:
            for b in sp:
                for op in OPS:
                    self._check(op, a, b, T16)


if __name__ == "__main__":
    unittest.main()
