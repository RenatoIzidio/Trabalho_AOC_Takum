# Zvtakum · Projeto Integrador Arquitetura e Organização de Computadores

**Engenharia Eletrônica · UFMG · 2026/02**  
**Autores:** Luiz Fernando Esteves Soares, Renato Pedro Izidio Lopes e Tulio Cesar de Lima Miranda

Este repositório contém um simulador funcional em Python para **Zvtakum**, uma proposta de cinco instruções de aritmética Takum linear para a extensão vetorial RISC-V (RVV). O projeto implementa os formatos T8, T16 e T32 e avalia o erro numérico na multiplicação de matriz esparsa por vetor (SpMV).

## Executar o projeto

Baixe ou clone o repositório. Abra um terminal na pasta de `run_all.py` e use **Python 3.12**, versão utilizada na validação.

**1. Crie o ambiente virtual.**

```sh
python -m venv .venv
```

**2. Ative o ambiente** com o comando correspondente ao seu terminal.

| Terminal | Comando |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Linux ou macOS (Bash/Zsh) | `source .venv/bin/activate` |

**3. Instale as dependências e execute.**

```sh
python -m pip install -r requirements.txt
python run_all.py
```

O script executa **49 testes automatizados** e, se todos passarem, realiza o benchmark sobre **34 matrizes sintéticas**, com semente 42. Ao terminar, abra `spmv_error_distribution.html` no navegador para consultar o gráfico e os resultados por matriz.

Para uma demonstração curta das instruções vetoriais, execute `python rvv_takum.py`. Para repetir apenas o benchmark, execute `python spmv_benchmark.py`.

## Organização do código

| Fase | Implementação | Testes |
|---|---|---|
| 1. Codificação, arredondamento e aritmética Takum linear | [takum.py](takum.py) | [test_takum.py](test_takum.py) |
| 2. Simulação funcional das instruções vetoriais e execução de SpMV | [rvv_takum.py](rvv_takum.py) | [test_rvv_takum.py](test_rvv_takum.py) |
| 3. Geração das matrizes, comparação dos formatos e cálculo do erro | [spmv_benchmark.py](spmv_benchmark.py) | [test_spmv_benchmark.py](test_spmv_benchmark.py) |

[run_all.py](run_all.py) coordena as três fases. As dependências estão em [requirements.txt](requirements.txt).

## Resultados gerados

| Arquivo | Conteúdo |
|---|---|
| `spmv_error_distribution.html` | Relatório para consulta no navegador, com resumo e resultados por matriz |
| `spmv_ecdf.svg` | Gráfico da distribuição acumulada do erro relativo em norma 2 |
| `spmv_summary.csv` | Erro mediano, número de resultados finitos e número de casos estáveis por formato e kernel |
| `spmv_matrices.csv` | Família, dimensões, número de elementos não nulos e esparsidade de cada matriz |
| `spmv_results.csv` | Erro individual de cada combinação de matriz, formato e kernel |
| `spmv_rvv_metrics.csv` | Contagem de instruções modeladas, operações e estimativas de ciclos |
| `spmv_benchmark_results.json` | Dados consolidados, análise de registradores de controle e identificador da execução |

Para comparar **formatos numéricos**, use as linhas `scalar`, que seguem a mesma ordem de acumulação. As linhas `RVV` avaliam a política vetorial de Takum, com redução em árvore e precisão ampliada. O simulador soma os resultados entre blocos com aritmética racional em Python. Os ciclos são estimativas do modelo de quatro *lanes*.

## Validação adicional com a libtakum

O script [validate_libtakum.py](validate_libtakum.py) compara a implementação Python com a biblioteca C de referência. Essa verificação é opcional e independente de `run_all.py`. Para reproduzi-la em Linux, com Git, Make e um compilador C instalados, execute a partir da raiz do projeto:

```bash
git clone https://github.com/takum-arithmetic/libtakum.git
git -C libtakum checkout 776ac0ca7855f640100731ed8bbb0c7cf83b2d94
(cd libtakum && ./configure && make)
python validate_libtakum.py ./libtakum/libtakum.so.2.0.0
```
