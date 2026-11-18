import ctypes as c
import math
from fractions import Fraction
import random
import sys

import takum

# Usage: python validate_libtakum.py /absolute/path/libtakum.so.2.0.0
lib = c.CDLL(sys.argv[1])
decode_wide = lib.takum64_to_float64
decode_wide.argtypes = [c.c_int64]
decode_wide.restype = c.c_double
failed = False
for n, integer in [(8, c.c_int8), (16, c.c_int16), (32, c.c_int32)]:
    rng = random.Random(42)
    mask = (1 << n) - 1
    def bind(name, ret, *args):
        f = getattr(lib, f'takum{n}_{name}')
        f.restype = ret
        f.argtypes = args
        return f
    sign = lambda x: integer(x).value
    enc = bind('from_float64', integer, c.c_double)
    dec = bind('to_float64', c.c_double, integer)
    def run(label, cases, local, reference, diagnostic=False):
        global failed
        count = errors = 0
        examples = []
        for case in cases:
            a, b = local(*case), reference(*case)
            count += 1
            if a != b and not (math.isnan(a) and math.isnan(b)):
                errors += 1
                if len(examples) < 3:
                    examples.append((case,a,b))
        if errors and not diagnostic:
            failed = True
        print(n, label, count, errors, examples, flush=True)
    edge = [0,1,2,(1<<(n-1))-2,(1<<(n-1))-1,1<<(n-1),(1<<(n-1))+1,mask-1,mask]
    patterns = sorted(set(edge+[rng.randrange(mask+1) for _ in range(20000)])) if n>8 else list(range(256))
    run('decode', ((x,) for x in patterns), lambda x: takum.to_float(x,n), lambda x: dec(sign(x)))
    values = [0.,-0.,math.inf,-math.inf,math.nan,1.,-1.]
    values += [math.ldexp(rng.uniform(-1, 1),rng.randint(-300,300)) for _ in range(20000)]
    run('encode', ((x,) for x in values), lambda x: takum.from_float(x,n), lambda x: enc(x)&mask)
    # Widen the encoded midpoint before decoding. The widened encoding
    # contains at most n+1 significant bits, so binary64 is exact here.
    boundary_cases = []
    one_ulp_cases = []
    for x in patterns:
        if x in (0, (1 << (n-1))-1, 1 << (n-1), mask):
            continue
        wide = c.c_int64(((x << 1) | 1) << (63 - n)).value
        threshold = Fraction.from_float(decode_wide(wide))
        delta = abs(threshold) * Fraction(1, 1 << (n + 2))
        boundary_cases.extend(((threshold-delta, x), (threshold, x+(x&1)), (threshold+delta, x+1)))
        mid = float(threshold)
        one_ulp_cases.extend(((math.nextafter(mid,-math.inf), x), (math.nextafter(mid,math.inf), x+1)))
    run('exact boundary', boundary_cases, lambda x,_: takum.from_float(x,n), lambda _,expected: expected)
    run('libtakum at tie', boundary_cases[1::3], lambda x,_: enc(float(x))&mask, lambda _,expected: expected)
    run('Python at one binary64 ULP', one_ulp_cases, lambda x,_: takum.from_float(x,n), lambda _,expected: expected)
    run('libtakum at one binary64 ULP (diagnostic)', one_ulp_cases, lambda x,_: enc(x)&mask, lambda _,expected: expected, diagnostic=True)
    pairs = [(x,y) for x in patterns for y in patterns] if n==8 else [(x,y) for x in edge for y in edge]+[(rng.choice(patterns),rng.choice(patterns)) for _ in range(20000)]
    for label, local in [('addition',takum.add),('subtraction',takum.sub),('multiplication',takum.mul)]:
        oracle = bind(label, integer, integer, integer)
        run(label, pairs, lambda x,y: local(x,y,n), lambda x,y: oracle(sign(x),sign(y))&mask)

raise SystemExit(int(failed))
