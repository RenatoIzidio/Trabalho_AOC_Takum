#!/usr/bin/env python3
"""
rvv_takum.py - Simulador Funcional de Instrucoes Vetoriais RISC-V (RVV 1.0)
com Extensao Zvtakum (Aritmetica Takum Linear T8 / T16 / T32).

Projeto Integrador AOC (UFMG, 2026/02) - Fases 2 e 3:
  5.2  Extensao do Conjunto de Instrucoes RISC-V (Zvtakum):
       - vtadd.vv: Adicao vetorial elemento a elemento em formato Takum (equiv vfadd.vv)
       - vtsub.vv: Subtracao vetorial elemento a elemento em formato Takum (equiv vfsub.vv)
       - vtmul.vv: Multiplicacao vetorial elemento a elemento em formato Takum (equiv vfmul.vv)
       - vtdot.vv: Produto escalar vetorial com acumulo em precisao ampliada (equiv vdot.* / vfdot.*)
       - vncvt.t.t: Conversao com reducao de largura de elemento (ex: T16 para T8) (equiv vfncvt.*)
  5.3  Suporte a execucao de kernels de algebra linear esparsa (SpMV y = A * x),
       contagem de instrucoes e analise de complexidade de CSRs.

Referencias:
  - UFMG DELT: Roteiro do Projeto Integrador AOC 2026/02.
  - L. Hunhold, "Streamlining SIMD ISA Extensions with Takum Arithmetic: A Case Study on Intel AVX10.2", arXiv:2503.14067 (2025).
  - L. Hunhold, "Beating Posits at Their Own Game: Takum Arithmetic" (2024).
  - RISC-V "V" Vector Extension Specification Version 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import takum as tk


# ==============================================================================
# Definicoes de Estado e Configuracao do RVV
# ==============================================================================

SUPPORTED_SEW = (8, 16, 32)      # T8, T16, T32
DEFAULT_VLEN = 256              # 256 bits (padrao de processadores modernos como C906/SpacemiT)


@dataclass
class VType:
    sew: int = 16       # Selected Element Width (8, 16 ou 32 bits)
    lmul: int = 1       # Register Grouping (1, 2, 4 ou 8)
    vta: bool = False   # Tail Agnostic (False = undisturbed)
    vma: bool = False   # Mask Agnostic (False = undisturbed)

    def encode(self) -> int:
        """Codifica o vtype em formato inteiro padrao RISC-V."""
        if self.sew not in SUPPORTED_SEW or self.lmul not in (1, 2, 4, 8):
            raise ValueError("vtype fora do subconjunto modelado")
        sew_enc = {8: 0, 16: 1, 32: 2}[self.sew]
        lmul_enc = {1: 0, 2: 1, 4: 2, 8: 3}.get(self.lmul, 0)
        return (int(self.vma) << 7) | (int(self.vta) << 6) | (sew_enc << 3) | lmul_enc

    @classmethod
    def decode(cls, val: int) -> "VType":
        sew_map = {0: 8, 1: 16, 2: 32, 3: 64}
        lmul_map = {0: 1, 1: 2, 2: 4, 3: 8}
        vma = bool((val >> 7) & 1)
        vta = bool((val >> 6) & 1)
        if val < 0 or val >> 8 or ((val >> 3) & 7) > 2 or (val & 7) > 3:
            raise ValueError("vtype fora do subconjunto modelado")
        sew = sew_map[(val >> 3) & 7]
        lmul = lmul_map.get(val & 0b111, 1)
        return cls(sew=sew, lmul=lmul, vta=vta, vma=vma)


@dataclass
class PerformanceCounters:
    """Contadores de metricas de execucao."""
    instructions_retired: int = 0
    vector_instructions: int = 0
    scalar_instructions: int = 0
    vector_element_ops: int = 0
    memory_load_bytes: int = 0
    memory_store_bytes: int = 0
    csr_accesses: int = 0
    cycles_estimated: int = 0

    def reset(self) -> None:
        self.instructions_retired = 0
        self.vector_instructions = 0
        self.scalar_instructions = 0
        self.vector_element_ops = 0
        self.memory_load_bytes = 0
        self.memory_store_bytes = 0
        self.csr_accesses = 0
        self.cycles_estimated = 0


# ==============================================================================
# Processador Vetorial RISC-V com Extensao Zvtakum
# ==============================================================================

class RVVState:
    """
    Estado do processador RISC-V com registradores vetoriais e suporte a Takum.
    Estima ciclos por lotes de elementos e niveis de reducao.
    """

    def __init__(self, vlen: int = DEFAULT_VLEN, num_lanes: int = 4):
        if vlen % 64 != 0 or vlen < 64:
            raise ValueError("VLEN deve ser multiplo de 64 e >= 64 bits")
        if not isinstance(num_lanes, int) or num_lanes < 1:
            raise ValueError("numero de lanes invalido")
        self.vlen = vlen
        self.vlenb = vlen // 8
        self.num_lanes = num_lanes  # Numero de unidades SIMD / Lanes operando em paralelo (ex: 4 lanes)

        # 32 registradores vetoriais (armazenados como bytes/inteiros)
        # Cada registrador v[i] armazena vlen bits
        self.v = [bytearray(self.vlenb) for _ in range(32)]

        # 32 registradores escalares x0..x31 (inteiros de 64 bits)
        self.x = [0] * 32

        # Registradores de Controle e Estado (CSRs)
        self.vtype = VType(sew=16, lmul=1)
        self.vl: int = 0

        # Memoria linear simulada (dicionario de bytes)
        self.memory: dict[int, int] = {}

        # Contadores de desempenho
        self.perf = PerformanceCounters()
        self.last_parallel_trace: list[str] = []

    # --------------------------------------------------------------------------
    # Gestao de Registradores Escalares e CSRs
    # --------------------------------------------------------------------------
    def get_x(self, reg: int) -> int:
        if not 0 <= reg < 32:
            raise ValueError("registrador escalar invalido")
        if reg == 0:
            return 0
        return self.x[reg] & 0xFFFFFFFFFFFFFFFF

    def set_x(self, reg: int, val: int) -> None:
        if not 0 <= reg < 32:
            raise ValueError("registrador escalar invalido")
        if reg != 0:
            self.x[reg] = val & 0xFFFFFFFFFFFFFFFF

    @property
    def vlmax(self) -> int:
        """VLMAX = (VLEN / SEW) * LMUL."""
        return (self.vlen // self.vtype.sew) * self.vtype.lmul

    def vsetvli(self, rd: int, rs1_val: int, sew: int, lmul: int = 1) -> int:
        """
        Instrucao vsetvli rd, rs1, vtypei
        Configura o tamanho do elemento (SEW) e agrupamento (LMUL),
        e calcula o comprimento do vetor ativo (vl) com base no AVL (rs1_val).
        """
        if rs1_val < 0 or not 0 <= rd < 32:
            raise ValueError("AVL ou rd invalido")
        if sew not in SUPPORTED_SEW:
            raise ValueError(f"SEW={sew} nao suportado na extensao Zvtakum (suporte: {SUPPORTED_SEW})")
        if lmul not in (1, 2, 4, 8):
            raise ValueError(f"LMUL={lmul} nao suportado")

        self.vtype = VType(sew=sew, lmul=lmul)
        vlmax = self.vlmax

        # Calculo do vl de acordo com a especificacao RVV 1.0:
        if rs1_val <= vlmax:
            self.vl = rs1_val
        elif rs1_val < 2 * vlmax:
            self.vl = (rs1_val + 1) // 2
        else:
            self.vl = vlmax

        self.set_x(rd, self.vl)
        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.csr_accesses += 1
        self.perf.cycles_estimated += 1
        return self.vl

    # --------------------------------------------------------------------------
    # Acesso a Elementos dos Registradores Vetoriais
    # --------------------------------------------------------------------------
    def _group(self, reg, width):
        if width not in SUPPORTED_SEW:
            raise ValueError("EEW invalido")
        emul = Fraction(self.vtype.lmul * width, self.vtype.sew)
        count = max(1, math.ceil(emul))
        if emul > 8 or reg < 0 or reg + count > 32 or reg % count:
            raise ValueError("grupo vetorial invalido")
        return range(reg, reg + count)

    def _location(self, reg, idx, width):
        self._group(reg, width)
        if not 0 <= idx < self.vlmax:
            raise IndexError("indice vetorial fora de VLMAX")
        step, offset = divmod(idx * (width // 8), self.vlenb)
        return reg + step, offset

    def _operands(self, vd, vm, *sources):
        if vm not in (0, 1) or (vm == 0 and vd == 0):
            raise ValueError("mascara ou destino invalido")
        if not 0 <= self.vl <= self.vlmax:
            raise ValueError("vl invalido")
        for reg in sources:
            self._group(reg, self.vtype.sew)

    def read_scalar_result(self, reg, width):
        if not 0 <= reg < 32 or width not in SUPPORTED_SEW:
            raise ValueError("resultado escalar invalido")
        return int.from_bytes(self.v[reg][:width // 8], "little")

    def get_element(self, reg: int, idx: int, sew: int | None = None) -> int:
        """Le o elemento 'idx' do registrador vetorial 'reg' na largura SEW."""
        sew = sew or self.vtype.sew
        bytes_per_elem = sew // 8
        reg, offset = self._location(reg, idx, sew)
        raw = self.v[reg][offset:offset + bytes_per_elem]
        return int.from_bytes(raw, byteorder="little")

    def set_element(self, reg: int, idx: int, val: int, sew: int | None = None) -> None:
        """Escreve o elemento 'idx' no registrador vetorial 'reg' na largura SEW."""
        sew = sew or self.vtype.sew
        bytes_per_elem = sew // 8
        reg, offset = self._location(reg, idx, sew)
        mask = (1 << sew) - 1
        raw = (val & mask).to_bytes(bytes_per_elem, byteorder="little")
        self.v[reg][offset:offset + bytes_per_elem] = raw

    def read_vector(self, reg: int, length: int | None = None, sew: int | None = None) -> list[int]:
        """Le uma lista de inteiros brutos dos primeiros 'length' elementos do registrador."""
        sew = sew or self.vtype.sew
        length = self.vl if length is None else length
        self._group(reg, sew)
        if not 0 <= length <= self.vlmax:
            raise IndexError("comprimento vetorial invalido")
        return [self.get_element(reg, i, sew) for i in range(length)]

    def write_vector(self, reg: int, elements: list[int], sew: int | None = None) -> None:
        """Escreve uma lista de elementos no registrador vetorial."""
        sew = sew or self.vtype.sew
        self._group(reg, sew)
        if len(elements) > self.vlmax:
            raise IndexError("vetor excede VLMAX")
        for i, val in enumerate(elements):
            self.set_element(reg, i, val, sew)

    def is_masked(self, idx: int, vm: int) -> bool:
        """
        Verifica se o elemento 'idx' esta ativo de acordo com o bit vm.
        vm = 1: instrucao desmascarada (sempre executa).
        vm = 0: predicao habilitada usando o registrador v0 como mascara.
        """
        if vm == 1:
            return True
        byte_idx = idx // 8
        bit_idx = idx % 8
        if byte_idx >= self.vlenb:
            return False
        return bool((self.v[0][byte_idx] >> bit_idx) & 1)

    # --------------------------------------------------------------------------
    # Memoria: Carga e Armazenamento (Load / Store / Indexed Gather)
    # --------------------------------------------------------------------------
    def mem_write_bytes(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self.memory[addr + i] = b

    def mem_read_bytes(self, addr: int, count: int) -> bytes:
        return bytes(self.memory.get(addr + i, 0) for i in range(count))

    def vle(self, vd: int, base_addr: int, vm: int = 1) -> None:
        """Vector Load Unit-Stride (vle8.v / vle16.v / vle32.v)."""
        self._operands(vd, vm, vd)
        sew = self.vtype.sew
        bytes_elem = sew // 8
        for i in range(self.vl):
            if self.is_masked(i, vm):
                elem_addr = base_addr + i * bytes_elem
                val = int.from_bytes(self.mem_read_bytes(elem_addr, bytes_elem), byteorder="little")
                self.set_element(vd, i, val, sew)
                self.perf.memory_load_bytes += bytes_elem
        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += max(1, math.ceil(self.vl / self.num_lanes))

    def vse(self, vs3: int, base_addr: int, vm: int = 1) -> None:
        """Vector Store Unit-Stride (vse8.v / vse16.v / vse32.v)."""
        self._operands(None, vm, vs3)
        sew = self.vtype.sew
        bytes_elem = sew // 8
        for i in range(self.vl):
            if self.is_masked(i, vm):
                elem_addr = base_addr + i * bytes_elem
                val = self.get_element(vs3, i, sew)
                self.mem_write_bytes(elem_addr, val.to_bytes(bytes_elem, byteorder="little"))
                self.perf.memory_store_bytes += bytes_elem
        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += max(1, math.ceil(self.vl / self.num_lanes))

    def vluxei(self, vd: int, base_addr: int, vs2: int, index_sew: int, vm: int = 1) -> None:
        """
        Vector Indexed (Gather) Load: vd[i] = memory[base_addr + vs2[i]].
        Instrucao fundamental para produtos esparsos (SpMV).
        """
        data_sew = self.vtype.sew
        self._operands(vd, vm, vd)
        dest = self._group(vd, data_sew)
        indices = self._group(vs2, index_sew)
        overlap = bool(set(dest).intersection(indices))
        legal_overlap = (data_sew == index_sew or
                         (data_sew < index_sew and vd == vs2) or
                         (data_sew > index_sew and indices.stop == dest.stop and
                          self.vtype.lmul * index_sew >= data_sew))
        if overlap and not legal_overlap:
            raise ValueError("sobreposicao entre destino e indices")
        offsets = self.read_vector(vs2, sew=index_sew)
        data_bytes = data_sew // 8
        for i in range(self.vl):
            if self.is_masked(i, vm):
                idx = offsets[i]
                elem_addr = base_addr + idx
                val = int.from_bytes(self.mem_read_bytes(elem_addr, data_bytes), byteorder="little")
                self.set_element(vd, i, val, data_sew)
                self.perf.memory_load_bytes += data_bytes
        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += max(2, self.vl)

    def vtmv_v_t(self, vd: int, val_takum: int) -> None:
        """Broadcast de um escalar Takum para todos os elementos ativos de vd."""
        self._operands(vd, 1, vd)
        sew = self.vtype.sew
        for i in range(self.vl):
            self.set_element(vd, i, val_takum, sew)
        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += 1

    # ==========================================================================
    # 5.2 INSTRUCOES PROPOSTAS DA EXTENSAO Zvtakum (Tabela 1 do PDF)
    # Modelagem explicita de Lanes Vetoriais Paralelas e Arvore de Reducao
    # ==========================================================================

    def _organize_parallel_lane_execution(self, inst_name: str, elem_op_fn: Callable[[int], None]) -> None:
        """
        Organiza a execucao dos elementos vetoriais em lotes paralelos de acordo com o
        numero de Lanes SIMD (self.num_lanes). Cada lote ocupa um ciclo no modelo funcional.
        """
        self.last_parallel_trace = []

        num_cycles = math.ceil(self.vl / self.num_lanes)
        for cycle in range(num_cycles):
            lane_tasks = []
            for lane_id in range(self.num_lanes):
                elem_idx = cycle * self.num_lanes + lane_id
                if elem_idx < self.vl:
                    lane_tasks.append(f"Lane {lane_id}: elem[{elem_idx}]")
                    elem_op_fn(elem_idx)
            self.last_parallel_trace.append(f"Ciclo {cycle+1} ({inst_name}): " + " | ".join(lane_tasks))

        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += max(1, num_cycles)

    def vtadd_vv(self, vd: int, vs2: int, vs1: int, vm: int = 1) -> None:
        """
        vtadd.vv vd, vs2, vs1, vm
        Adicao vetorial elemento a elemento em formato Takum (equivalente RVV: vfadd.vv).
        Executada em paralelo atraves das Lanes Vetoriais.
        """
        self._operands(vd, vm, vd, vs2, vs1)
        sew = self.vtype.sew

        def do_elem(i: int) -> None:
            if self.is_masked(i, vm):
                op2 = self.get_element(vs2, i, sew)
                op1 = self.get_element(vs1, i, sew)
                res = tk.add(op2, op1, sew)
                self.set_element(vd, i, res, sew)
                self.perf.vector_element_ops += 1

        self._organize_parallel_lane_execution("vtadd.vv", do_elem)

    def vtsub_vv(self, vd: int, vs2: int, vs1: int, vm: int = 1) -> None:
        """
        vtsub.vv vd, vs2, vs1, vm
        Subtracao vetorial elemento a elemento em formato Takum (equivalente RVV: vfsub.vv).
        Executada em paralelo atraves das Lanes Vetoriais.
        """
        self._operands(vd, vm, vd, vs2, vs1)
        sew = self.vtype.sew

        def do_elem(i: int) -> None:
            if self.is_masked(i, vm):
                op2 = self.get_element(vs2, i, sew)
                op1 = self.get_element(vs1, i, sew)
                res = tk.sub(op2, op1, sew)
                self.set_element(vd, i, res, sew)
                self.perf.vector_element_ops += 1

        self._organize_parallel_lane_execution("vtsub.vv", do_elem)

    def vtmul_vv(self, vd: int, vs2: int, vs1: int, vm: int = 1) -> None:
        """
        vtmul.vv vd, vs2, vs1, vm
        Multiplicacao vetorial elemento a elemento em formato Takum (equivalente RVV: vfmul.vv).
        Executada em paralelo atraves das Lanes Vetoriais.
        """
        self._operands(vd, vm, vd, vs2, vs1)
        sew = self.vtype.sew

        def do_elem(i: int) -> None:
            if self.is_masked(i, vm):
                op2 = self.get_element(vs2, i, sew)
                op1 = self.get_element(vs1, i, sew)
                res = tk.mul(op2, op1, sew)
                self.set_element(vd, i, res, sew)
                self.perf.vector_element_ops += 1

        self._organize_parallel_lane_execution("vtmul.vv", do_elem)

    def vtdot_vv(self, vd: int, vs2: int, vs1: int, vm: int = 1, widened: bool = True) -> int:
        """
        vtdot.vv vd, vs2, vs1, vm
        Produto escalar vetorial com acumulo em precisao ampliada (equivalente RVV: vdot.* / vfdot.*).

        Implementa a arquitetura de ARVORE DE REDUCAO PARALELA (Parallel Reduction Tree):
          Estagio 1: Multiplicacoes elemento a elemento executadas em paralelo nas Lanes.
          Estagio 2: Arvore binaria de somadores paralelos (complexidade temporal profundidade logaritmica; ciclos dependem das lanes).
        """
        self._operands(vd, vm, vs2, vs1)
        sew = self.vtype.sew
        acc_sew = (sew * 2) if (widened and sew * 2 <= 32) else sew
        def round_node(value):
            # T32 ampliado usa acumulacao racional exata no modelo funcional.
            if widened and sew == 32:
                return value
            return tk.to_fraction(tk.from_float(value, acc_sew), acc_sew)

        old = self.read_scalar_result(vd, acc_sew)
        self.last_parallel_trace = []
        if self.vl == 0:
            self.perf.instructions_retired += 1
            self.perf.vector_instructions += 1
            self.perf.cycles_estimated += 1
            return old

        # 1. Multiplicacoes parciais paralelas nas lanes
        partial_products: list[Fraction | None] = []
        num_cycles = math.ceil(self.vl / self.num_lanes) if self.vl > 0 else 1

        for cycle in range(num_cycles):
            lane_tasks = []
            for lane_id in range(self.num_lanes):
                i = cycle * self.num_lanes + lane_id
                if i < self.vl:
                    if self.is_masked(i, vm):
                        op2 = self.get_element(vs2, i, sew)
                        op1 = self.get_element(vs1, i, sew)
                        self.perf.vector_element_ops += 1
                        lane_tasks.append(f"Lane {lane_id}: op2[{i}]*op1[{i}]")

                        if op2 == tk.nar_pattern(sew) or op1 == tk.nar_pattern(sew):
                            partial_products.append(None)
                        else:
                            q2 = tk.to_fraction(op2, sew)
                            q1 = tk.to_fraction(op1, sew)
                            if q2 is None or q1 is None:
                                partial_products.append(None)
                            else:
                                partial_products.append(round_node(q2 * q1))
                    else:
                        partial_products.append(Fraction(0))
            if lane_tasks:
                self.last_parallel_trace.append(f"Estagio 1 (Mul Paralela - Ciclo {cycle+1}): " + " | ".join(lane_tasks))

        # 2. Arvore de Reducao Paralela (Parallel Binary Adder Tree)
        current_layer = partial_products[:]
        tree_level = 1
        tree_cycles = 0

        while len(current_layer) > 1:
            next_layer = []
            level_tasks = []
            tree_cycles += math.ceil((len(current_layer) // 2) / self.num_lanes)
            for k in range(0, len(current_layer), 2):
                if k + 1 < len(current_layer):
                    a, b = current_layer[k], current_layer[k+1]
                    if a is None or b is None:
                        res = None
                    else:
                        res = round_node(a + b)
                    level_tasks.append(f"Somador {k//2}: ({k} + {k+1})")
                    self.perf.vector_element_ops += 1
                    next_layer.append(res)
                else:
                    # Elemento impar passa direto para o proximo nivel
                    next_layer.append(current_layer[k])

            self.last_parallel_trace.append(f"Estagio 2 (Arvore de Reducao - Nivel {tree_level}): " + " | ".join(level_tasks))
            current_layer = next_layer
            tree_level += 1

        final_val = current_layer[0] if current_layer else Fraction(0)
        if final_val is None:
            res_bits = tk.nar_pattern(acc_sew)
        else:
            res_bits = tk.from_float(final_val, acc_sew)

        # Grava no primeiro elemento de vd na largura apropriada
        self.v[vd][:acc_sew // 8] = res_bits.to_bytes(acc_sew // 8, "little")

        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += (num_cycles + tree_cycles)
        return res_bits

    def vncvt_t_t(self, vd: int, vs2: int, vm: int = 1) -> None:
        """
        vncvt.t.t vd, vs2, vm
        Conversao com reducao de largura de elemento (narrowing conversion, ex: T16 para T8, T32 para T16).
        Equivalente RVV: vfncvt.*

        Origem (vs2): largura 2 * SEW.
        Destino (vd): largura SEW.
        Realiza decodificacao exata e re-arredondamento (round-to-nearest, ties-to-even) para a largura menor.
        """
        dst_sew = self.vtype.sew
        src_sew = dst_sew * 2
        if src_sew > 32:
            raise ValueError(f"Narrowing conversion nao suportada para SEW_dst={dst_sew} (exigiria SEW_src={src_sew})")

        self._operands(vd, vm, vd)
        if vd != vs2 and set(self._group(vd, dst_sew)).intersection(self._group(vs2, src_sew)):
            raise ValueError("sobreposicao invalida no narrowing")
        for i in range(self.vl):
            if self.is_masked(i, vm):
                src_val = self.get_element(vs2, i, src_sew)
                if src_val == tk.nar_pattern(src_sew):
                    dst_val = tk.nar_pattern(dst_sew)
                elif src_val == 0:
                    dst_val = 0
                else:
                    # Decodifica de forma exata usando _unpack e recodifica em dst_sew
                    kind, neg, N, E = tk._unpack(src_val, src_sew)
                    dst_val = tk._round_pack(neg, N, E, dst_sew)
                self.set_element(vd, i, dst_val, dst_sew)
                self.perf.vector_element_ops += 1

        self.perf.instructions_retired += 1
        self.perf.vector_instructions += 1
        self.perf.cycles_estimated += max(1, math.ceil(self.vl / self.num_lanes))


# ==============================================================================
# Kernel de Algebra Linear Esparsa: SpMV (y = A * x) no Simulador RVV
# ==============================================================================

class SparseMatrixCSR:
    """Representacao de matriz esparsa em formato Compressed Sparse Row (CSR)."""

    def __init__(self, nrows: int, ncols: int, row_ptr: list[int], col_ind: list[int], values: list[float]):
        if (nrows < 0 or ncols < 0 or len(row_ptr) != nrows + 1 or
                row_ptr[0] != 0 or row_ptr[-1] != len(values) or len(col_ind) != len(values)
                or any(a > b for a,b in zip(row_ptr,row_ptr[1:]))
                or any(not isinstance(i, int) or not 0 <= i < ncols for i in col_ind)
                or any(not isinstance(i, int) for i in row_ptr)):
            raise ValueError("estrutura CSR invalida")
        self.nrows = nrows
        self.ncols = ncols
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.values = values
        self.nnz = len(values)

    @classmethod
    def from_dense(cls, mat: list[list[float]]) -> "SparseMatrixCSR":
        nrows = len(mat)
        ncols = len(mat[0]) if nrows > 0 else 0
        row_ptr = [0]
        col_ind = []
        values = []
        for r in range(nrows):
            for c in range(ncols):
                val = mat[r][c]
                if val != 0.0:
                    values.append(val)
                    col_ind.append(c)
            row_ptr.append(len(values))
        return cls(nrows, ncols, row_ptr, col_ind, values)


def execute_spmv_rvv_takum(
    matrix: SparseMatrixCSR,
    x_dense: list[float],
    sew: int = 16,
    vlen: int = DEFAULT_VLEN,
) -> tuple[list[float], PerformanceCounters]:
    """
    Executa a multiplicacao de matriz por vetor esparsa (SpMV y = A * x)
    utilizando diretamente as instrucoes da extensao Zvtakum do RVV.

    Para cada linha i:
      y[i] = sum_{k = row_ptr[i] ... row_ptr[i+1]-1} A[k] * x[col_ind[k]]
    Utiliza:
      - vsetvli para ajustar o comprimento ativo de acordo com o numero de elementos nao nulos da linha
      - Carga dos valores A[k] no registrador v1 (vle)
      - Gather dos elementos correspondentes de x usando indices col_ind no registrador v2 (vluxei)
      - Produto escalar ampliado com vtdot.vv para o acumulador v3
      - Armazenamento do resultado escalar em y[i]
    """
    if len(x_dense) != matrix.ncols:
        raise ValueError("dimensoes incompativeis")
    rvv = RVVState(vlen=vlen)
    bytes_elem = sew // 8
    idx_sew = 16 if max(0, matrix.ncols - 1) * bytes_elem < 65536 else 32
    if max(0, matrix.ncols - 1) * bytes_elem >= 2**32:
        raise ValueError("deslocamento excede 32 bits")


    # Alocacao de memoria simulada para o vetor x
    x_base_addr = 0x10000
    for c_idx, val in enumerate(x_dense):
        val_tk = tk.from_float(val, sew)
        rvv.mem_write_bytes(x_base_addr + c_idx * bytes_elem, val_tk.to_bytes(bytes_elem, byteorder="little"))

    y_results: list[float] = []

    for r in range(matrix.nrows):
        start = matrix.row_ptr[r]
        end = matrix.row_ptr[r + 1]
        row_nnz = end - start

        if row_nnz == 0:
            y_results.append(0.0)
            continue

        # Acumula pedacos de tamanho VLMAX
        row_accum_frac = Fraction(0)
        invalid = False
        curr_idx = start

        while curr_idx < end:
            remaining = end - curr_idx
            vl = rvv.vsetvli(rd=1, rs1_val=remaining, sew=sew, lmul=1)

            # Prepara os chunks de valores A e indices de colunas
            chunk_vals = [tk.from_float(matrix.values[curr_idx + j], sew) for j in range(vl)]
            chunk_cols = [bytes_elem * matrix.col_ind[curr_idx + j] for j in range(vl)]

            # Escreve indices em v4 (para gather) e valores em v1
            rvv.write_vector(reg=4, elements=chunk_cols, sew=idx_sew)
            rvv.write_vector(reg=1, elements=chunk_vals, sew=sew)

            # Realiza gather de x[col_ind] a partir da memoria para v2
            # vluxei v2, (x_base_addr), v4
            rvv.vluxei(vd=2, base_addr=x_base_addr, vs2=4, index_sew=idx_sew, vm=1)

            # Executa o dot product vetorial ampliado: v3 = dot(v1, v2)
            rvv.vtdot_vv(vd=3, vs2=1, vs1=2, vm=1, widened=True)

            acc_sew = (sew * 2) if sew * 2 <= 32 else sew
            dot_bits = rvv.read_scalar_result(3, acc_sew)
            dot_frac = tk.to_fraction(dot_bits, acc_sew)
            if dot_frac is None:
                invalid = True
            else:
                row_accum_frac += dot_frac

            curr_idx += vl

        # Converte o acumulado final de volta para o formato de precisao da operacao
        final_bits = tk.nar_pattern(sew) if invalid else tk.from_float(row_accum_frac, sew)
        y_results.append(tk.to_float(final_bits, sew))

    return y_results, rvv.perf


# ==============================================================================
# Analise de Complexidade Arquitetural de CSRs (Takum vs IEEE 754)
# ==============================================================================

def analyze_csr_complexity() -> dict[str, Any]:
    """
    Gera um comparativo detalhado da complexidade do registrador de status e
    controle de ponto flutuante (fcsr) entre o padrao IEEE 754 e a extensao Takum.
    """
    return {
        "RVV": "vl e vtype controlam o comprimento e o tipo vetorial. Operacoes FP podem usar frm e acumular fflags em fcsr.",
        "Zvtakum": "Cinco operacoes propostas (soma, subtracao, multiplicacao, dot e narrowing) compartilham SEW. Arredondamento fixo e NaR dispensam configurar frm nestas operacoes. vl e vtype continuam em uso.",
        "comparacao": "Soma, subtracao e produto admitem analogos vfadd, vfsub e vfmul. Uma reducao ampliada pode exigir varias instrucoes RVV, dependendo do formato e da politica. Sem backend RVV IEEE equivalente nao se estima reducao quantitativa.",
        "contadores": "O simulador conta chamadas modeladas de vsetvli, vluxei e vtdot por bloco. Preparacao de operandos em Python, loop escalar e acesso fisico nao entram nessa contagem. Sem medidas de area, energia ou speedup."
    }


if __name__ == "__main__":
    print("=== Teste Basico do Simulador RVV Zvtakum ===")
    rvv = RVVState(vlen=128)
    vl = rvv.vsetvli(rd=1, rs1_val=8, sew=16, lmul=1)
    print(f"Configurado: SEW=16, LMUL=1, VLMAX={rvv.vlmax}, VL={vl}")

    # Inicializa registradores vetoriais com valores Takum T16
    v1_vals = [tk.from_float(x, 16) for x in [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, -3.0]]
    v2_vals = [tk.from_float(x, 16) for x in [2.0, 0.5, 1.0, 2.0, 4.0, 2.0, 0.5, 2.0]]
    rvv.write_vector(1, v1_vals)
    rvv.write_vector(2, v2_vals)

    # Testa vtadd.vv
    rvv.vtadd_vv(vd=3, vs2=1, vs1=2)
    res_add = [tk.to_float(b, 16) for b in rvv.read_vector(3)]
    print(f"vtadd.vv: {res_add}")

    # Testa vtmul.vv
    rvv.vtmul_vv(vd=4, vs2=1, vs1=2)
    res_mul = [tk.to_float(b, 16) for b in rvv.read_vector(4)]
    print(f"vtmul.vv: {res_mul}")

    # Testa vtdot.vv (produto escalar ampliado)
    rvv.vtdot_vv(vd=5, vs2=1, vs1=2, widened=True)
    res_dot_bits = rvv.read_scalar_result(5, 32)
    print(f"vtdot.vv (acumulado em T32): {tk.to_float(res_dot_bits, 32)}")

    print(f"Instrucoes retiradas: {rvv.perf.instructions_retired}, Operacoes de elemento: {rvv.perf.vector_element_ops}")
