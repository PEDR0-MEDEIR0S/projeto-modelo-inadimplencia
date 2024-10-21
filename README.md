# projeto-modelo-inadimplencia
Este projeto tem como objetivo construir um modelo preditivo que calcula a probabilidade de inadimplência de clientes recorrentes para novos pedidos de crédito. O modelo é treinado e testado com dados anonimizados e parcialmente sintéticos fornecidos pela empresa, conforme descrito nas instruções do case.

# Estrutura do Projeto
O projeto é organizado da seguinte maneira:

Instruções_Case_Datarisk.pdf: Contém as instruções e requisitos da empresa sobre o projeto, para a execução e entregas finais.

base_cadastral.csv: Contém dados cadastrais fixos de cada cliente.

base_info.csv: Traz informações mensais sobre a renda e o número de funcionários dos clientes.

base_pagamentos_desenvolvimento.csv e base_pagamentos_teste.csv: Contêm informações sobre as transações de crédito, incluindo os prazos de vencimento e os pagamentos realizados.

# Objetivo
O desafio proposto é calcular a probabilidade de inadimplência para clientes recorrentes, definindo a inadimplência como um atraso de 5 dias ou mais no pagamento de uma nota de crédito. O modelo desenvolvido gera uma previsão probabilística de inadimplência para cada transação presente na base de teste, sendo essa previsão avaliada com base na performance e na qualidade do código.

# Metodologia

## 1. Análise Exploratória dos Dados
Inicialmente, realizei a análise exploratória das três bases de dados fornecidas para compreender a estrutura e qualidade dos dados.
Foram identificadas e tratadas possíveis inconsistências, como dados ausentes e incoerências comuns em datasets reais.

## 2. Pré-processamento dos Dados
Base cadastral: Realizei a limpeza de dados categóricos como o domínio de e-mails e o segmento industrial.
Base info: Efetuei a criação de variáveis agregadas para melhor capturar o comportamento temporal dos clientes.
Base pagamentos: Aqui, o foco foi no cálculo da variável de atraso (diferença entre a data de vencimento e a data de pagamento), gerando a variável-alvo INADIMPLENTE.

## 3. Feature Engineering
Combinei as três bases de dados para formar uma base unificada, utilizando o ID_CLIENTE e o SAFRA_REF como chaves de junção.
Criei novas variáveis derivadas das informações cadastrais e financeiras, como a relação entre renda e valor da nota de crédito, e também a taxa de inadimplência por segmento de mercado.

## 4. Construção e Treinamento do Modelo
Para a construção do modelo, utilizei o XGBoost Classifier (XGBClassifier), dado o seu desempenho superior em tarefas de classificação.
A coluna SAFRA_REF foi tratada apenas como identificador, não sendo incluída no treinamento do modelo.
A métrica de avaliação principal utilizada foi o AUC-ROC, devido à necessidade de um bom equilíbrio entre precisão e recall para a previsão de inadimplência.

## 5. Avaliação e Validação
A base de teste (base_pagamentos_teste.csv) foi utilizada para validação do modelo, gerando as previsões finais.
O desempenho foi avaliado internamente antes da geração dos resultados.

## 6. Entrega
O arquivo final gerado contém as colunas ID_CLIENTE, SAFRA_REF e a probabilidade de inadimplência calculada na coluna INADIMPLENTE.

# Resultados
As previsões de inadimplência foram salvas no arquivo previsoes.csv, contendo as seguintes colunas:
ID_CLIENTE: Identificador do cliente.
SAFRA_REF: Mês de referência da transação.
INADIMPLENTE: Probabilidade predita de inadimplência.

# Considerações Finais
O modelo desenvolvido demonstrou uma performance robusta ao lidar com inconsistências nos dados e ao gerar previsões de inadimplência. Potenciais melhorias podem incluir a utilização de dados adicionais, bem como o ajuste fino do modelo para otimização de resultados específicos.
