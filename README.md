# Zvtakum · Projeto Integrador AOC

**Engenharia Eletrônica · UFMG · 2026/02**  
Luiz Fernando Esteves Soares, Renato Pedro Izidio Lopes e Tulio Cesar de Lima Miranda

Este pacote reúne o código-fonte das três fases, os testes e os resultados do experimento de multiplicação de matriz esparsa por vetor (SpMV).

## Executar

Extraia o ZIP e abra um terminal na pasta que contém este README e o arquivo run_all.py. A versão utilizada foi Python 3.12. Crie e ative um ambiente virtual:

    python -m venv .venv

No Windows PowerShell, execute `.\.venv\Scripts\Activate.ps1`. No Linux ou macOS, execute `source .venv/bin/activate`. Em seguida:

    python -m pip install -r requirements.txt
    python run_all.py

O comando executa 49 testes e regenera os resultados de 34 matrizes com a semente 42. Para uma demonstração curta das instruções vetoriais, execute `python rvv_takum.py`. Abra `spmv_error_distribution.html` no navegador para explorar os resultados já incluídos.

## Conteúdo

| Parte | Implementação | Verificação e documentação |
|---|---|---|
| Aritmética Takum linear T8, T16 e T32 | takum.py | test_takum.py, VALIDACAO_FASE_1.md |
| Simulador funcional da extensão vetorial | rvv_takum.py | test_rvv_takum.py, FASE_02.md |
| Benchmark de SpMV, formatos e métricas | spmv_benchmark.py | test_spmv_benchmark.py, FASE_03.md |

A comparação adicional com a biblioteca C de referência usa `validate_libtakum.py`. Para reproduzi-la em Linux, obtenha a versão indicada abaixo, compile a biblioteca e informe o caminho do arquivo compartilhado:

    git clone https://github.com/takum-arithmetic/libtakum.git
    git -C libtakum checkout 776ac0ca7855f640100731ed8bbb0c7cf83b2d94
    (cd libtakum && ./configure && make)
    python validate_libtakum.py ./libtakum/libtakum.so.2.0.0

A execução principal em Python é independente dessa comparação externa.

## Resultados

O JSON consolida as métricas, a análise de registradores de controle e a identificação da execução. As quatro tabelas CSV contêm o resumo por formato, a descrição das matrizes, os erros individuais e os contadores do simulador. O SVG e o HTML mostram a distribuição acumulada do erro. As estimativas de ciclos descrevem o modelo funcional com quatro lanes.
