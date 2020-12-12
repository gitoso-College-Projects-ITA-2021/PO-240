# PO-240: Tópicos avançados de Pesquisa Operacional - Introdução à IA

Essa branch contém os códigos relativos ao modelo de redes neurais implementado.
Há 3 arquivos principais:
    - preprocessamento.py : lê os dados originais, processando-os para o treinamento
    - treinamento.py      : lê o arquivo preprocessado.csv gerado e roda o algoritmo de aprendizagem
    - avaliacao.py        : lê os pesos da rede salva e faz a regressão sobre todos os arquivos public2019XX.csv

Além desses, há os arquivos auxiliares:
    - dataset.py : funções auxiliares para leitura, processamento e separação dos dados
    - model.py   : funções auxiliares para a geração dos modelos
