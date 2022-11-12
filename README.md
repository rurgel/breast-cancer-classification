<h1>Classificação de tumores de câncer de mama por imagens histológicas</h1>


<h2>Introdução</h2>
Este repositório propõe uma aplicação de redes neurais convolucionais para diferenciar tumores benignos e malignos a partir das imagens resultantes da biópsia das mamas. Os dados dos cortes histológicos podem ser obtidos publicamente a partir deste <a href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/">Link</a>.

<br>
<p style='background-color:#FF0;color:#F00;font=weight:'><b>Aviso Legal: Esse programa não se destina a substituir o aconselhamento médico profissional. Em hipótese nenhuma ele deve ser considerado como qualquer referência para aconselhamento de saúde.</b></p>
<br>

<h2>Performance</h2>

A performance obtida na base de dados de validação na versão 0.0.4 foi:
|     | PREDITCTION |           |
|-----------|-------------|-----------|
|   **ACTUAL**        | Benign      | Malignant |
| Benign    | <span style="color:green">270</span>         | <span style="color:red">191</span>       |
| Malignant | <span style="color:red">61</span>          | <span style="color:green">1102</span>      |

<h2>Instalação</h2>

Para instalar esse pacote a partir do gerenciador de pacotes `pip`, abra a pasta raiz deste projeto no terminal e em seguida digite o seguinte comando:

```bash
pip install -e .
```

<h2>Utilização</h2>

Para a utilização é necessário <a href="https://drive.google.com/file/d/12ozVh6HPWWqmTvV8PhhhnFOSxVesGANK/view?usp=share_link">baixar o modelo pré-treinado</a> e mover os arquivos para a pasta models ou então baixar os dados de treino e realizar o treinamento da base de dados.

Assim é possível realizar as predições, para tanto basta executar o comando `breast-cancer` no terminal, acompanhado do caminho do arquivo de imagem a ser analisado:

```bash
breast-cancer <NOME-ARQUIVO>
```


<h2>Referências</h2>
Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., <b>A Dataset for Breast Cancer Histopathological Image Classification</b>, <i>IEEE Transactions on Biomedical Engineering (TBME)</i>, 63(7):1455-1462, 2016.