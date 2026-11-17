# Análise Psicométrica de Itens

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white) ![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green?style=flat-square) ![Status](https://img.shields.io/badge/status-em%20constru%C3%A7%C3%A3o-orange?style=flat-square)

> Pipeline reprodutível para avaliar a qualidade de itens de múltipla escolha a partir de uma matriz de respostas.

## 📌 Visão geral

Recebe as respostas dos participantes e o gabarito e produz, para cada item, os indicadores clássicos de qualidade, a análise de distratores e um relatório consolidado da prova. Pensado para simulados no formato de exames de larga escala (ex.: ENAMED).

## ✨ Funcionalidades

- Índice de dificuldade (p) e discriminação (D, ponto-bisserial)
- Análise de distratores com curvas por grupo de desempenho
- Confiabilidade da prova (KR-20 / alfa de Cronbach) e efeito de remoção de itens
- Classificação automática dos itens (manter, revisar, descartar)
- Calibração por Teoria de Resposta ao Item (modelos 1PL/2PL/3PL)
- Relatório final em HTML/PDF

## 🗂️ Estrutura do repositório

```text
analise-psicometrica-itens/
├── src/            # funções de cálculo (TCT, TRI, distratores)
├── exemplos/       # gerador de dados sintéticos e notebook de demonstração
├── relatorios/     # modelos de relatório
├── tests/
├── LICENSE
└── README.md
```

## 🚀 Como usar

```bash
python exemplos/gerar_dados_exemplo.py --participantes 300 --itens 100
python -m src.analise --respostas exemplos/respostas.csv --gabarito exemplos/gabarito.csv
```

## 🔒 Dados e privacidade

Este repositório **não contém dados reais** de estudantes, docentes ou instituições.
Todos os exemplos usam **dados sintéticos** gerados pelo próprio projeto
(`exemplos/gerar_dados_exemplo`), com identificadores fictícios (`Estudante 001`, `Item 001`…).
Os itens de prova usados nos exemplos são inventados para fins de demonstração.

## 🛣️ Roadmap

- [ ] Gerador de dados sintéticos
- [ ] Indicadores da TCT
- [ ] Análise de distratores
- [ ] Calibração TRI
- [ ] Relatório automatizado

## 📄 Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE).
