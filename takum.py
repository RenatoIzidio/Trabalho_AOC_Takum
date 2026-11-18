#!/usr/bin/env python3
"""
takum.py - Simulador funcional de aritmetica Takum linear (T8 / T16 / T32).

Projeto Integrador AOC (UFMG, 2026/02) - Fase 1:
  5.1.1  Extracao de S, D, R, C e F de um padrao de n bits.
  5.1.2  Adicao, subtracao e multiplicacao para T8, T16 e T32.

Formato (L. Hunhold, "Beating Posits at Their Own Game: Takum Arithmetic",
Definicao 8 - takum linear). Padrao de n bits, do MSB para o LSB:

        S | D | R (3 bits) | C (r bits) | F (p bits)

    r = 7 - uint(R)                 se D = 0      r = uint(R)             se D = 1
    c = -2^(r+1) + 1 + uint(C)      se D = 0      c = 2^r - 1 + uint(C)   se D = 1
    p = max(0, n - r - 5)  (em n < 12, C pode conter ghost bits)
    f = uint(F) / 2^p                              (fracao, em [0, 1))
    e = (-1)^S * (c + S)                           (expoente efetivo)

    valor = [(1 - 3S) + f] * 2^e                   (caso geral)
    0   = 000...0        NaR = 100...0             (S=1, resto zero)

 * S=0: valor = (1 + f) * 2^c ;  S=1: valor = (-2 + f) * 2^-(c+1).
   Nao ha complemento de dois na decodificacao: o sinal ja entra na formula.
 * Negar um takum = complemento de dois do padrao; a ordem dos padroes
   (como inteiros com sinal) coincide com a ordem dos valores (NaR e o menor).
 * Padroes com n < 12 bits sao completados com zeros a direita ("ghost bits").
   Por isso o decodificador so precisa olhar os 12 bits mais significativos
   para obter S, D, R e C, para qualquer n (e a estrutura comum do enunciado).

Arredondamento (takum linear, de acordo com libtakum/src/util.c):
  * arredondamento para o mais proximo, com empate para o par (ties to even);
  * bits descartados (guard + sticky) decidem se ha empate verdadeiro;
  * numeros nao nulos saturam nos extremos, nunca em zero ou NaR.

As operacoes usam inteiros de precisao arbitraria: resultados intermediarios
sao exatos e arredondados uma unica vez para o formato Takum de destino.
Entradas decimais podem ser convertidas como Fraction, sem passar por float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

T8, T16, T32 = 8, 16, 32

__all__ = [
    "T8", "T16", "T32", "Fields", "Takum", "decode_fields", "to_fraction",
    "to_float", "from_float", "add", "sub", "mul", "neg",
    "nar_pattern", "max_pos", "min_pos", "max_neg", "min_neg",
]


# --------------------------------------------------------------------------
# Utilidades de padroes especiais
# --------------------------------------------------------------------------
def _check_width(n: int) -> None:
    """O formato admite n >= 5; T8/T16/T32 sao os tamanhos da disciplina."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("largura n deve ser um inteiro")
    if n < 5:
        raise ValueError("largura n deve ser >= 5")


def _mask(n: int) -> int:
    _check_width(n)
    return (1 << n) - 1


def _check(bits: int, n: int) -> None:
    _check_width(n)
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("padrao bits deve ser um inteiro")
    if not 0 <= bits <= _mask(n):
        raise ValueError(f"padrao 0x{bits:X} nao cabe em {n} bits")


def nar_pattern(n: int) -> int:
    """NaR = 1000...0."""
    _check_width(n)
    return 1 << (n - 1)


def max_pos(n: int) -> int:
    """Maior positivo = 0111...1."""
    return (1 << (n - 1)) - 1


def min_pos(n: int) -> int:
    """Menor positivo = 000...01."""
    _check_width(n)
    return 1


def max_neg(n: int) -> int:
    """Negativo de maior modulo = 1000...01."""
    _check_width(n)
    return (1 << (n - 1)) + 1


def min_neg(n: int) -> int:
    """Negativo de menor modulo = 111...1."""
    return _mask(n)


# --------------------------------------------------------------------------
# 5.1.1  Extracao dos campos S, D, R, C, F
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fields:
    n: int
    bits: int
    S: int          # bit de sinal
    D: int          # bit de direcao
    R: int          # campo de regime (3 bits, como lido)
    r: int          # valor do regime = numero de bits de caracteristica
    C: int          # bits de caracteristica (r bits; ghost bits = 0)
    c: int          # caracteristica
    F: int          # bits de fracao
    p: int          # numero de bits de fracao presentes (>= 0)
    e: int          # expoente efetivo (-1)^S (c + S)
    kind: str       # "zero" | "nar" | "num"

    @property
    def c_bits_present(self) -> int:
        """Bits de C realmente presentes (em T8 alguns ficam de fora: ghost)."""
        return max(0, min(self.r, self.n - 5))

    @property
    def ghost_bits(self) -> int:
        return self.r - self.c_bits_present

    @property
    def f(self) -> Fraction:
        return Fraction(self.F, 1 << self.p)

    def bitstring(self) -> str:
        """Padrao separado em 'S D RRR C F' ('-' = campo vazio)."""
        s = format(self.bits, f"0{self.n}b")
        cb = self.c_bits_present
        parts = [s[0], s[1], s[2:5], s[5:5 + cb] or "-", s[5 + cb:] or "-"]
        return " ".join(parts)

    def describe(self) -> str:
        head = f"T{self.n}  0x{self.bits:0{(self.n + 3) // 4}X}  {self.bitstring()}"
        if self.kind != "num":
            return f"{head}\n  -> {'zero' if self.kind == 'zero' else 'NaR'}"
        ghost = f" (+{self.ghost_bits} ghost)" if self.ghost_bits else ""
        return (
            f"{head}\n"
            f"  S={self.S}  D={self.D}  R={self.R:03b} (r={self.r})  "
            f"C={self.C} (c={self.c}){ghost}  F={self.F} (p={self.p}, f={float(self.f):.6g})\n"
            f"  e={self.e}  valor={to_float(self.bits, self.n):.10g}"
        )


def decode_fields(bits: int, n: int) -> Fields:
    """Extrai S, D, R, C, F (e derivados) de um padrao de n bits."""
    _check(bits, n)
    # Janela de 12 bits: S, D, R e ate 7 bits de C estao sempre aqui.
    top = (bits >> (n - 12)) if n >= 12 else (bits << (12 - n))
    S = (top >> 11) & 1
    D = (top >> 10) & 1
    R = (top >> 7) & 0b111
    r = R if D else 7 - R
    C = (top >> (7 - r)) & ((1 << r) - 1)
    p = max(0, n - 5 - r)
    F = bits & ((1 << p) - 1)
    c = ((1 << r) - 1 + C) if D else (-(1 << (r + 1)) + 1 + C)
    e = -(c + 1) if S else c
    if bits == 0:
        kind = "zero"
    elif bits == nar_pattern(n):
        kind = "nar"
    else:
        kind = "num"
    return Fields(n, bits, S, D, R, r, C, c, F, p, e, kind)


# --------------------------------------------------------------------------
# Decodificacao para valor exato: (-1)^neg * N * 2^E,  N > 0 inteiro
# --------------------------------------------------------------------------
def _unpack(bits: int, n: int) -> tuple[str, bool, int, int]:
    f = decode_fields(bits, n)
    if f.kind != "num":
        return f.kind, False, 0, 0
    if f.S == 0:
        N = (1 << f.p) + f.F            # (1 + f) * 2^p
    else:
        N = (1 << (f.p + 1)) - f.F      # (2 - f) * 2^p
    return "num", bool(f.S), N, f.e - f.p


def _dyadic(neg: bool, N: int, E: int) -> Fraction:
    v = Fraction(N << E) if E >= 0 else Fraction(N, 1 << -E)
    return -v if neg else v


def to_fraction(bits: int, n: int) -> Fraction | None:
    """Valor exato (Fraction). NaR -> None."""
    kind, neg, N, E = _unpack(bits, n)
    if kind == "nar":
        return None
    if kind == "zero":
        return Fraction(0)
    return _dyadic(neg, N, E)


def to_float(bits: int, n: int) -> float:
    """Valor como float64 (exato para T8/T16/T32). NaR -> nan."""
    kind, neg, N, E = _unpack(bits, n)
    if kind == "nar":
        return math.nan
    if kind == "zero":
        return 0.0
    v = math.ldexp(N, E)
    return -v if neg else v


# --------------------------------------------------------------------------
# Codificacao com arredondamento (Algoritmo 1 do artigo)
# --------------------------------------------------------------------------
def _round_pack(neg: bool, N: int, E: int, n: int) -> int:
    """Codifica +/- N*2**E (N>0), arredondando para o par mais proximo.

    O desempate usa todos os bits descartados, nao somente o bit guard.
    """
    _check_width(n)
    if isinstance(N, bool) or not isinstance(N, int) or N <= 0:
        raise ValueError("N deve ser um inteiro positivo")
    if isinstance(E, bool) or not isinstance(E, int):
        raise TypeError("expoente E deve ser inteiro")
    Q = N.bit_length() - 1
    h = E + Q                       # |x| = (1 + g) * 2^h, g = G / 2^Q
    G = N - (1 << Q)

    sat_hi = max_neg(n) if neg else max_pos(n)
    sat_lo = min_neg(n) if neg else min_pos(n)
    # Fora da faixa dinamica (|x| >= 2^255  ou  |x| <= 2^-255): satura.
    if h >= 255:
        return sat_hi
    if h < -255 or (h == -255 and G == 0):
        return sat_lo

    # Caracteristica c e fracao F (Algoritmo 4 do artigo).
    if not neg:
        S, c, F, FQ = 0, h, G, Q
    elif G == 0:
        S, c, F, FQ = 1, -h, 0, 0
    else:
        S, c, F, FQ = 1, -h - 1, (1 << Q) - G, Q

    if c >= 0:
        D = 1
        r = (c + 1).bit_length() - 1
        R, C = r, c - (1 << r) + 1
    else:
        D = 0
        r = (-c).bit_length() - 1
        R, C = 7 - r, c + (1 << (r + 1)) - 1

    # Padrao sem perdas: S D R C F  (comprimento 5 + r + FQ).
    head = (((S << 1) | D) << 3) | R
    full = (((head << r) | C) << FQ) | F
    L = 5 + r + FQ

    # Round-to-nearest, ties-to-even: analisar guard E todos os bits
    # descartados (sticky), como em libtakum/src/util.c.
    if L <= n:
        T = full << (n - L)
    else:
        discarded = L - n
        truncated = full >> discarded
        remainder = full & ((1 << discarded) - 1)
        halfway = 1 << (discarded - 1)
        increment = remainder > halfway or (remainder == halfway and (truncated & 1) == 1)
        T = truncated + int(increment)
    T &= _mask(n)

    # Nunca arredonda para 0 nem para NaR.
    if T == 0:
        return sat_lo
    if T == nar_pattern(n):
        return sat_hi
    return T


def _from_rational_exact(q: Fraction, n: int) -> int:
    """Codifica um racional exato usando os campos Takum e guard/sticky.

    Para n < 12, pode faltar espaco ate mesmo para todos os bits de C.
    Nesse caso, os bits faltantes de C e a fracao f tambem participam
    da decisao de arredondamento. Isto evita converter decimais a float64.
    """
    neg = q < 0
    magnitude = abs(q)
    numerator, denominator = magnitude.numerator, magnitude.denominator
    h = numerator.bit_length() - denominator.bit_length()
    # Ajustar a estimativa de floor(log2(|q|)) usando somente inteiros.
    if h >= 0:
        if numerator < denominator << h:
            h -= 1
    elif numerator << (-h) < denominator:
        h -= 1

    if h >= 255:
        return max_neg(n) if neg else max_pos(n)
    if h < -255 or (h == -255 and numerator * (1 << 255) == denominator):
        return min_neg(n) if neg else min_pos(n)

    normalised = magnitude / (1 << h) if h >= 0 else magnitude * (1 << -h)
    g = normalised - 1                   # |q| = (1 + g) 2**h, 0 <= g < 1
    if not neg:
        S, c, f = 0, h, g
    elif g == 0:
        S, c, f = 1, -h, Fraction(0)
    else:
        S, c, f = 1, -h - 1, 1 - g

    if c >= 0:
        D = 1
        r = (c + 1).bit_length() - 1
        R, C = r, c - (1 << r) + 1
    else:
        D = 0
        r = (-c).bit_length() - 1
        R, C = 7 - r, c + (1 << (r + 1)) - 1
    head = (((S << 1) | D) << 3) | R
    p = n - 5 - r
    if p >= 0:
        scaled = f * (1 << p)
        fraction_bits = scaled.numerator // scaled.denominator
        encoded = (head << (n - 5)) | (C << p) | fraction_bits
        remainder = scaled - fraction_bits
    else:
        missing = -p
        encoded = (head << (n - 5)) | (C >> missing)
        remainder = Fraction(C & ((1 << missing) - 1), 1 << missing) + f / (1 << missing)

    # Arredondamento no espaco de codificacao Takum (guard/sticky).
    # Em regioes com ghost bits, este ponto medio nao precisa coincidir
    # com a media aritmetica dos valores reais de dois Takums vizinhos.
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and (encoded & 1)):
        encoded += 1
    encoded &= _mask(n)
    if encoded == 0:
        return min_neg(n) if neg else min_pos(n)
    if encoded == nar_pattern(n):
        return max_neg(n) if neg else max_pos(n)
    return encoded


def from_float(x: int | float | Fraction | str, n: int) -> int:
    """Codifica um numero em Takum com nearest, ties-to-even.

    Aceita int, float, Fraction e string decimal. Strings sao parseadas
    como racionais EXATOS (ex.: '0.1' = 1/10). Float preserva o valor
    binario ja aproximado pelo Python. NaN/inf do tipo float viram NaR.
    """
    _check_width(n)
    if isinstance(x, float) and not math.isfinite(x):
        return nar_pattern(n)
    q = Fraction(x)
    if q == 0:
        return 0
    denominator = q.denominator
    if denominator & (denominator - 1):
        return _from_rational_exact(q, n)
    return _round_pack(q < 0, abs(q.numerator), -(denominator.bit_length() - 1), n)


# --------------------------------------------------------------------------
# 5.1.2  Aritmetica: neg, add, sub, mul
# --------------------------------------------------------------------------
def neg(a: int, n: int) -> int:
    """Negacao = complemento de dois do padrao (0 e NaR sao pontos fixos)."""
    _check(a, n)
    return (-a) & _mask(n)


def add(a: int, b: int, n: int) -> int:
    ka, na, Na, Ea = _unpack(a, n)
    kb, nb, Nb, Eb = _unpack(b, n)
    if ka == "nar" or kb == "nar":
        return nar_pattern(n)
    if ka == "zero":
        return b
    if kb == "zero":
        return a
    E = min(Ea, Eb)                              # alinhamento exato
    va = (Na << (Ea - E)) * (-1 if na else 1)
    vb = (Nb << (Eb - E)) * (-1 if nb else 1)
    s = va + vb
    if s == 0:
        return 0
    return _round_pack(s < 0, abs(s), E, n)


def sub(a: int, b: int, n: int) -> int:
    return add(a, neg(b, n), n)


def mul(a: int, b: int, n: int) -> int:
    ka, na, Na, Ea = _unpack(a, n)
    kb, nb, Nb, Eb = _unpack(b, n)
    if ka == "nar" or kb == "nar":
        return nar_pattern(n)
    if ka == "zero" or kb == "zero":
        return 0
    return _round_pack(na != nb, Na * Nb, Ea + Eb, n)


# --------------------------------------------------------------------------
# Classe de conveniencia (demos, relatorio, benchmarks da Fase 3)
# --------------------------------------------------------------------------
class Takum:
    """Takum de n bits com operadores +, -, * e unario -."""

    __slots__ = ("bits", "n")

    def __init__(self, bits: int, n: int):
        _check(bits, n)
        self.bits, self.n = bits, n

    @classmethod
    def from_float(cls, x, n: int) -> "Takum":
        return cls(from_float(x, n), n)

    def _other(self, o) -> int:
        if isinstance(o, Takum):
            if o.n != self.n:
                raise ValueError("operandos com larguras diferentes")
            return o.bits
        return from_float(o, self.n)

    def __add__(self, o): return Takum(add(self.bits, self._other(o), self.n), self.n)
    def __sub__(self, o): return Takum(sub(self.bits, self._other(o), self.n), self.n)
    def __rsub__(self, o): return Takum(sub(self._other(o), self.bits, self.n), self.n)
    def __mul__(self, o): return Takum(mul(self.bits, self._other(o), self.n), self.n)
    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return Takum(neg(self.bits, self.n), self.n)
    def __float__(self): return to_float(self.bits, self.n)
    def __eq__(self, o): return isinstance(o, Takum) and (self.n, self.bits) == (o.n, o.bits)
    def __hash__(self): return hash((self.n, self.bits))

    @property
    def fields(self) -> Fields:
        return decode_fields(self.bits, self.n)

    def __repr__(self) -> str:
        w = (self.n + 3) // 4
        if self.bits == nar_pattern(self.n):
            return f"T{self.n}(0x{self.bits:0{w}X} = NaR)"
        return f"T{self.n}(0x{self.bits:0{w}X} = {float(self):.10g})"


# --------------------------------------------------------------------------
# Linha de comando
# --------------------------------------------------------------------------
def _parse_operand(s: str, n: int) -> int:
    if s.lower().startswith("0x"):
        bits = int(s, 16)
        _check(bits, n)
        return bits
    # Preserva a entrada de especiais via linha de comando.
    if s.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return from_float(float(s), n)
    # Nao usa float(s): o valor decimal e convertido sem perda intermediaria.
    return from_float(s, n)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Simulador Takum linear (T8/T16/T32)")
    sub_ = ap.add_subparsers(dest="cmd", required=True)

    d = sub_.add_parser("decode", help="mostra S, D, R, C, F de um padrao (hex) ou numero")
    d.add_argument("value", help="0x... (padrao de bits) ou numero decimal")
    d.add_argument("-n", type=int, default=16, choices=[8, 16, 32])

    c = sub_.add_parser("calc", help="opera dois valores: add | sub | mul")
    c.add_argument("op", choices=["add", "sub", "mul"])
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-n", type=int, default=16, choices=[8, 16, 32])

    args = ap.parse_args(argv)
    n = args.n
    if args.cmd == "decode":
        print(decode_fields(_parse_operand(args.value, n), n).describe())
    else:
        a, b = _parse_operand(args.a, n), _parse_operand(args.b, n)
        r = {"add": add, "sub": sub, "mul": mul}[args.op](a, b, n)
        print(f"A = {Takum(a, n)!r}")
        print(f"B = {Takum(b, n)!r}")
        print(f"{args.op}(A, B) -> {Takum(r, n)!r}")
        print(decode_fields(r, n).describe())


if __name__ == "__main__":
    main()
