# %%
"""
Importação de bibliotecas necessárias

Aqui importamos bibliotecas essenciais para manipulação de dados, visualização gráfica,
modelagem preditiva e outras operações estatísticas. Cada biblioteca será utilizada em 
momentos distintos, dependendo das necessidades de análise ou modelagem.
Todas as bibliotecas essenciais:
- Manipulação de dados (`pandas`, `numpy`)
- Visualização de gráficos (`matplotlib`, `seaborn`, `plotly`)
- Modelagem preditiva e avaliação de modelos (`scikit-learn`, `xgboost`)
- Análises estatísticas e econométricas (`scipy`, `statsmodels`, `prince`)
- Análises específicas (e.g., SHAP para interpretabilidade de modelos)
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from scipy.stats import chi2_contingency
import prince
from xgboost import XGBClassifier
import plotly.graph_objects as go
import webbrowser
import networkx as nx
import matplotlib.cm as cm
import statsmodels.api as sm
import shap
from datetime import datetime

def imprimir_hora_atual():
    hora_atual = datetime.now()
    print("Hora atual:", hora_atual.strftime("%H:%M:%S"))
    
# %%
"""
Carregamento das bases de dados

Neste passo, carregamos as bases de dados 'cadastral', 'informações' e 'pagamentos'. 
Essas bases contêm dados de clientes, características de pagamento e outras informações relevantes 
para a análise.

**Observação**: Os dados foram previamente anonimizados para evitar a exposição de informações sensíveis.
"""
base_cadastral = pd.read_csv('base_cadastral.csv')
base_info = pd.read_csv('base_info.csv')
base_pagamentos = pd.read_csv('base_pagamentos_desenvolvimento.csv')
base_pagamentos_teste = pd.read_csv('base_pagamentos_teste.csv')

# %%
"""
Unificação dos datasets

As três bases foram unificadas em uma tabela principal chamada 'df', que servirá como a tabela 
de contingência. Essa tabela é fundamental para a análise posterior e para a construção do modelo 
preditivo.
"""
df = base_pagamentos.merge(base_cadastral, on='ID_CLIENTE').merge(base_info, on=['ID_CLIENTE', 'SAFRA_REF'])
# Exibindo a estrutura básica do DataFrame para entender sua composição
df.info()
df.describe()

# %%
"""
Limpeza e preparação dos dados

1. **Criação da coluna `INADIMPLENTES`**: Indica inadimplência com base no atraso de 
pagamento superior a 5 dias.
2. **Remoção de dados ausentes**: Linhas com valores faltantes nas colunas essenciais 
para modelagem são eliminadas.
3. **Tratamento da coluna `DDD`**: Remoção de caracteres especiais.
4. **Transformação da coluna `DATA_VENCIMENTO`**: Separação em colunas de ano, mês e 
dia para facilitar futuras análises temporais.
"""

# Criando a coluna 'INADIMPLENTES' com base em atraso superior a 5 dias
df['INADIMPLENTES'] = (pd.to_datetime(df['DATA_PAGAMENTO']) - pd.to_datetime(df['DATA_VENCIMENTO'])).dt.days >= 5
df['INADIMPLENTES'] = df['INADIMPLENTES'].astype(int)

# Removendo linhas com dados ausentes
df.dropna(subset=['VALOR_A_PAGAR', 'TAXA', 'DDD', 'SEGMENTO_INDUSTRIAL', 'PORTE', 'NO_FUNCIONARIOS'], inplace=True)

# Removendo colunas desnecessárias
df.drop(columns=['FLAG_PF'], inplace=True, errors='ignore')

# Limpando caracteres especiais na coluna 'DDD'
df['DDD'] = df['DDD'].str.replace(r'\D', '', regex=True)

# Garantindo que o DataFrame não esteja vazio após a limpeza
if df.empty:
    raise ValueError("O DataFrame está vazio após o pré-processamento.")

# Transformando 'DATA_VENCIMENTO' em colunas separadas de ano, mês e dia
df['DATA_VENCIMENTO'] = pd.to_datetime(df['DATA_VENCIMENTO'], format='%Y-%m-%d')
df['ANO_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.year
df['MES_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.month
df['DIA_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.day

# Removendo a coluna original 'DATA_VENCIMENTO'
df.drop(columns=['DATA_VENCIMENTO'], inplace=True)

# Visualizando a distribuição de inadimplentes por ano de vencimento
plt.figure(figsize=(10, 6))
sns.countplot(x='ANO_VENCIMENTO', data=df, hue='INADIMPLENTES')
plt.title('Distribuição de Inadimplentes por Ano de Vencimento')
plt.show()

# Exibindo informações básicas do DataFrame após limpeza
df.info()
df.describe()

# %%
"""
Visualização 3D interativa dos dados.

Neste ponto, criamos alguns gráficos 3D interativo para explorar a relação entre 
inadimplência, localização (CEP_2_DIG) e valor a pagar entre outras variáveis, com
o objetivo de compreender melhor as informações sobre os dados.
"""
trace = go.Scatter3d(
    x=df['INADIMPLENTES'],
    y=df['CEP_2_DIG'],
    z=df['VALOR_A_PAGAR'],
    mode='markers',
    marker=dict(
        size=10,
        color='darkorchid',
        opacity=0.7,
    )
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
    plot_bgcolor='white',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        ),
        yaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        ),
        zaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        )
    )
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.update_layout(scene=dict(
    xaxis_title='INADIMPLENTES',
    yaxis_title='CEP_2_DIG',
    zaxis_title='VALOR_A_PAGAR'
))

plot_figure.write_html('temp_scatter3D.html')
webbrowser.open('temp_scatter3D.html')

# %%
"""
Visualização 3D interativa dos dados - Análise de correlações

Este bloco cria um gráfico de correlação (heatmap) para variáveis numéricas, facilitando 
a identificação de relações entre diferentes fatores e a inadimplência.
"""
# Selecionando as colunas numéricas e excluindo IDs e datas irrelevantes
columns_to_exclude = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_CADASTRO']
numeric_columns = df.select_dtypes(include=['number']).columns

columns_to_include = numeric_columns.difference(columns_to_exclude)
# Calculando a matriz de correlação
correlation_matrix = df[columns_to_include].corr()

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 8}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=11)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=11)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
plt.show()


columns_to_exclude = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_CADASTRO', 'ANO_VENCIMENTO', 'MES_VENCIMENTO', 'DIA_VENCIMENTO']
numeric_columns = df.select_dtypes(include=['number']).columns

# Filtrar colunas, removendo as que queremos ignorar
columns_to_include = numeric_columns.difference(columns_to_exclude)

# Calcular a matriz de correlação
correlation_matrix = df[columns_to_include].corr()

# Criar o grafo
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in correlation_matrix.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(correlation_matrix.columns):
    for j, variable2 in enumerate(correlation_matrix.columns):
        if i != j:
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.get_cmap('coolwarm_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 25 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                         edge_cmap=cmap, alpha=0.7)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Exibição do gráfico
plt.show()

modelo = sm.OLS.from_formula("INADIMPLENTES ~ VALOR_A_PAGAR + RENDA_MES_ANTERIOR + NO_FUNCIONARIOS + PORTE + TAXA", df).fit()

# Parâmetros do 'modelo'
modelo.summary()

# Cálculo do R² ajustado
r2_ajust = 1-((len(df.index)-1)/(len(df.index)-\
                                          modelo.params.count()))*\
    (1-modelo.rsquared)
r2_ajust # modo direto: modelo.rsquared_adj

"""
Analise OLS:
    
R-squared (R²):
Valor: 0.015
Interpretação: Apenas 1.5% da variação na variável dependente (INADIMPLENTES) é explicada pelo modelo. Isso sugere que o modelo tem um baixo poder preditivo e que outras variáveis não incluídas podem ser importantes.

F-statistic:
Valor: 158.0

Prob (F-statistic): 4.45e-200
Interpretação: O modelo é estatisticamente significativo (p < 0.05), o que indica que pelo menos uma das variáveis independentes é significativa na previsão da variável dependente.

Coeficientes:
Intercept: 0.0992 — O valor esperado de INADIMPLENTES quando todas as variáveis independentes são zero.
Variáveis categóricas (PORTE):
PORTE[T.MEDIO]: 0.0151 — Indica que, em comparação com a categoria de referência (presumivelmente "grande"), estar na categoria "médio" está associado a um aumento de 0.0151 em INADIMPLENTES.
PORTE[T.PEQUENO]: 0.0427 — Estar na categoria "pequeno" está associado a um aumento de 0.0427 em INADIMPLENTES.
VALOR_A_PAGAR: -4.534e-07 — Um aumento de uma unidade em VALOR_A_PAGAR está associado a uma diminuição de 0.0000004534 em INADIMPLENTES, o que é uma relação muito pequena.
RENDA_MES_ANTERIOR: -6.284e-08 — Um aumento de uma unidade em RENDA_MES_ANTERIOR está associado a uma diminuição de 0.00000006284 em INADIMPLENTES, sugerindo que um aumento na renda pode reduzir a inadimplência.
NO_FUNCIONARIOS: -8.004e-05 — Não é estatisticamente significativo (p = 0.139), portanto, não podemos concluir que essa variável tem um impacto.
TAXA: -0.0002 — Também não é estatisticamente significativo (p = 0.755), indicando que essa variável não tem um efeito relevante.

Formula do modelo:

Y^(INADIMPLENTES)
​
 = 0.0992+0.0151⋅PORTEMEDIO
​
 +0.0427⋅PORTEPEQUENO
​
 −4.534×10^−7 * VALOR_A_PAGAR
 
 −6.284×10^−8 * RENDA_MES_ANTERIOR
 
 −8.004×10^−5 * NO_FUNCIONARIOS
 
 −0.0002⋅TAXA
 .
 .
 .
 
"""

# %%
"""
Testando correlação das colunas categóricas com a variável alvo usando o 
test_chi_square.

Através deste teste, iremos obter p-values que indicam a relação entre as 
variáveis categóricas e a variável de referência 'INADIMPLENTES'.

O teste de Qui-quadrado nos ajuda a identificar quais variáveis categóricas têm uma 
associação estatisticamente significativa com a variável de referência, mas não 
estabelece causalidade. Variáveis com p-value acima de 0.05 serão removidas, pois não
mostram correlação relevante.

Algumas variáveis foram removidas devido à falta de relevância ou duplicidade de informações, como:
- 'SAFRA_REF' (apenas referência de amostra)
- 'DATA_PAGAMENTO' (foi objeto para calcular inadimplência, tornando dispensável)
- 'DOMINIO_EMAIL' (sem relação de causalidade sobre a váriavel alvo)
- 'DDD' (informação duplicada no CEP pois acompanha índices de localidade)
"""
def test_chi_square(data, target):
    p_values = {}
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Incluindo colunas de vencimento para análise
    categorical_columns += ['ANO_VENCIMENTO', 'MES_VENCIMENTO', 'DIA_VENCIMENTO']
    
    for col in categorical_columns:
        contingency_table = pd.crosstab(data[col], data[target])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:  # Garante que a tabela tem dados
            _, p_value, _, _ = chi2_contingency(contingency_table)
            p_values[col] = p_value
        else:
            p_values[col] = np.nan  # Se não houver dados, atribui NaN ao p-valor
    return p_values

# Indicando a coluna INADIMPLENTES para o teste
p_values = test_chi_square(df, 'INADIMPLENTES')

# Exibindo os p-valores
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])
print(p_values_df)

# Removendo colunas com p-valor maior que 0.05
cols_to_remove = p_values_df[p_values_df['P-Value'] > 0.05]['Feature'].tolist()
cols_to_remove = [col for col in cols_to_remove if col != 'INADIMPLENTES']  # Não remover INADIMPLENTES

# Selecionando colunas adicionais para remoção com justificativas
cols_to_remove.extend([
    'SAFRA_REF',  # Apenas referência do mês da amostra
    'DATA_PAGAMENTO',  # Já usada para calcular inadimplência
    'DOMINIO_EMAIL',  # Não influencia no pagamento
    'DDD'  # Duplicidade com CEP
])

# Removendo as colunas
df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

df.info()
df.describe()

# %%
"""
Análise exploratória dos dados.

Aqui visualizamos a distribuição de inadimplentes e a matriz de correlação entre 
as variáveis numéricas. Esta análise ajuda a entender padrões e possíveis associações.
"""
sns.countplot(x='INADIMPLENTES', data=df)
plt.title('Distribuição de Inadimplentes')
plt.show()

# Calculando a correlação apenas para colunas numéricas
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# %%
"""
Preparando os dados para o modelo.

Nesta etapa, estamos separando as variáveis independentes (X) da variável dependente (y), 
que é a que queremos prever. A variável "INADIMPLENTES" é nossa variável alvo, enquanto 
"ID_CLIENTE" é removida, pois não é uma característica que contribui para a previsão.

Depois, os dados são divididos em conjuntos de treinamento e teste usando a função 
`train_test_split`. O conjunto de treinamento é utilizado para ajustar o modelo, 
enquanto o conjunto de teste é reservado para avaliar a performance do modelo em 
dados que ele não viu antes. A divisão é feita com uma proporção de 80% para o 
conjunto de treinamento e 20% para o conjunto de teste.

Além disso configurei o pré-processamento dos dados para garantir que estejam na
forma adequada para o modelo. As variáveis numéricas são normalizadas usando o 
`StandardScaler`, que padroniza os dados para que tenham média zero e desvio padrão um. 
Isso é importante para algoritmos que dependem da escala dos dados. As variáveis 
categóricas são transformadas em variáveis dummies (ou one-hot encoding) com o 
`OneHotEncoder`, que cria colunas binárias para cada categoria, permitindo que o
modelo utilize essas informações corretamente.

Por fim, estamos criando um pipeline que integra o pré-processamento e o classificador 
(neste caso, um `RandomForestClassifier`). O uso de um pipeline facilita a aplicação 
de pré-processamento e ajuste do modelo em um único passo, tornando o código mais 
organizado e modular. Após a configuração, o pipeline é ajustado aos dados de treinamento.
"""
# Separando variáveis independentes (X) e dependente (y)
X = df.drop(['INADIMPLENTES', 'ID_CLIENTE', 'DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO'], axis=1)
y = df['INADIMPLENTES']

# Verificando se os dados de X e y estão corretos
if X.empty or y.empty:
    raise ValueError("X ou y estão vazios. Verifique os dados.")

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo as variáveis numéricas e categóricas para pré-processamento
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist() + ['ANO_VENCIMENTO', 'MES_VENCIMENTO', 'DIA_VENCIMENTO']

# Pipeline de pré-processamento e modelagem
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Ajustando o pipeline aos dados de treinamento
pipeline.fit(X_train, y_train)

# Avaliando o desempenho no conjunto de teste
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# %%
"""
Conclusão e próximos passos.

Após o ajuste do modelo, verificamos uma acurácia de X% (dependendo dos dados reais).
No entanto, a acurácia não deve ser a única métrica para avaliar este modelo. 
Devemos considerar métricas adicionais, como precisão, recall e F1-score, para entender 
melhor o comportamento do modelo em diferentes cenários de previsão.

Além disso, uma análise de feature importance pode ser feita para identificar as variáveis 
que mais influenciam as previsões do modelo. 

Próximos passos:
1. Implementar métricas adicionais (precisão, recall, F1-score).
2. Realizar tuning de hiperparâmetros do RandomForestClassifier usando GridSearchCV ou RandomizedSearchCV.
3. Realizar análise de feature importance.
"""
#X_test = X_test.drop(columns=["DATA_CADASTRO", "DATA_EMISSAO_DOCUMENTO"])
# X_test_filtered = X_test.drop(columns=["DATA_CADASTRO", "DATA_EMISSAO_DOCUMENTO"], errors='ignore')

# Cálculo de métricas adicionais
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Precisão: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

# Plotando feature importance
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(10))
plt.title('Top 10 Features Importantes')
plt.show()

# %% 
"""
Preparando os dados para ACM.

Nesta etapa, estamos iniciando na Análise de Correspondência Múltipla (ACM), que é uma 
técnica estatística utilizada para explorar a relação entre variáveis categóricas. 
Para isso, selecionamos apenas as variáveis categóricas do nosso DataFrame, pois a 
ACM requer esse tipo de dados para identificar padrões e associações.

Em seguida, estamos criando a Matriz Binária Z usando a função `get_dummies`. Essa matriz 
transforma nossas variáveis categóricas em um formato binário(dicotômico), onde cada 
categoria é representada por uma coluna separada. Isso é essencial para que a ACM 
possa ser realizada corretamente, permitindo a análise das interações entre as 
diferentes categorias.
"""
# Selecionando apenas as variáveis categóricas
categorical_data = df[categorical_features]

# Criando a Matriz Binária Z
Z = pd.get_dummies(categorical_data)

# %%
""" 
Validação Cruzada e Avaliação do Modelo.

Agora vamos melhorar a avaliação do modelo utilizando validação cruzada (cross-validation), 
que nos permite estimar de forma mais robusta a performance do modelo em diferentes partições 
do conjunto de dados. Isso ajuda a reduzir a variação entre diferentes divisões de treino e teste.

A seguir, implementamos a validação cruzada com 5 folds (KFold), e também armazenamos as 
métricas de avaliação (precisão, recall, F1-score e acurácia) para analisar a consistência
do modelo em todas as iterações.

# Importante: Para evitar erros, execute esta parte do script por trechos de linhas,
# separando cada trecho com uma linha em branco, em vez de utilizar células no Spyder.
"""
# Excecutar a próxima etapa pode demorar um pouco, de 5 a 15 minutos a depender da CPU
# Importante: Para evitar erros, execute esta parte do script por trechos de linhas,
# separando cada trecho com uma linha em branco, em vez de utilizar células no Spyder.
# Utilizando KFold para validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy')
print(f'Média de Acurácia na Validação Cruzada: {cv_scores.mean():.4f}')
print(f'Desvio padrão: {cv_scores.std():.4f}')

# Tuning de hiperparâmetros com GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
# Excecutar a próxima etapa pode demorar um pouco, de 5 a 15 minutos a depender da CPU
# Em teste, foi exigido um bom desempenho da CPU para que o processo fosse otimizado
# Importante: Para evitar erros, execute esta parte do script por trechos de linhas,
# separando cada trecho com uma linha em branco, em vez de utilizar células no Spyder.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Avaliação com os melhores parâmetros
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))
print(f'Acurácia com melhores parâmetros: {accuracy_score(y_test, y_pred_best):.4f}')

# %%
""" 
Matriz de Confusão e Curva ROC.

Para uma análise mais detalhada, criamos a matriz de confusão e calculamos a AUC 
(Área sob a Curva) da curva ROC, que é uma métrica relevante para modelos binários. 
A curva ROC nos permite avaliar a capacidade do modelo em distinguir entre as classes 
positivas e negativas (INADIMPLENTES e NÃO-INADIMPLENTES). 

# Importante: Para evitar erros, execute esta parte do script por trechos de linhas,
# separando cada trecho com uma linha em branco, em vez de utilizar células no Spyder.
"""
# Previsões sobre o conjunto de teste
y_pred = pipeline.predict(X_test)
# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()
# Importante: Para evitar erros, execute esta parte do script por trechos de linhas,
# separando cada trecho com uma linha em branco, em vez de utilizar células no Spyder.
# Curva ROC
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6)) 
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange') 
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Taxa de Falsos Positivos') 
plt.ylabel('Taxa de Verdadeiros Positivos') 
plt.title('Curva ROC') 
plt.legend() 
plt.show()

# %%
"""
Aprimoramento do Modelo: Tuning de Hiperparâmetros com GridSearchCV.

Agora, adicionamos uma etapa de tuning de hiperparâmetros utilizando GridSearchCV. 
Isso permite encontrar a melhor combinação de parâmetros do modelo RandomForest 
para maximizar o desempenho preditivo. Avaliamos múltiplas configurações, como o 
número de estimadores (n_estimators), a profundidade máxima das árvores (max_depth), 
entre outros. 
"""
# Definindo a grade de parâmetros
param_grid = { 'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [10, 20, None], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4] }

# Configurando o GridSearch
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# Excecutar a próxima etapa pode demorar um pouco, de 5 a 15 minutos a depender da CPU
# Realizando a busca pelos melhores parâmetros
grid_search.fit(X_train, y_train)

# Exibindo os melhores parâmetros encontrados
print(f"Melhores parâmetros: {grid_search.best_params_}")

# Avaliando o modelo com os melhores parâmetros
best_model = grid_search.best_estimator_ 
y_pred_best = best_model.predict(X_test) 
print(classification_report(y_test, y_pred_best)) 
print("Acurácia com melhores parâmetros:", accuracy_score(y_test, y_pred_best))

# %%
""" 
Importância das Variáveis (Feature Importance).

Uma análise fundamental para entender o comportamento do modelo RandomForest é a 
importância das variáveis. Aqui, identificamos as variáveis que mais influenciam 
as previsões de inadimplência, o que nos ajuda a tomar decisões mais informadas e 
justificadas.
"""
feature_importances = best_model.named_steps['classifier'].feature_importances_

# Obtenha os nomes das características após a transformação
feature_names = numeric_features + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())

# Crie o DataFrame de importâncias
importances_df = pd.DataFrame({
    'Variável': feature_names,
    'Importância': feature_importances
})

# Visualize a importância das variáveis
plt.figure(figsize=(10, 8))
sns.barplot(x='Importância', y='Variável', data=importances_df.sort_values(by='Importância', ascending=False))
plt.title('Importância das Variáveis - Random Forest')
plt.show()

# %%Interpretação do Modelo com SHAP
"""
Interpretação do Modelo com SHAP.

Utilizamos SHAP (SHapley Additive exPlanations) para explicar as previsões do modelo,
o que nos permite entender de forma mais detalhada o impacto de cada variável individual 
nas previsões. O SHAP nos fornece uma visão mais granular e transparente sobre o 
funcionamento interno do modelo.
"""
# Informações sobre o conjunto de dados Z
Z.info()

# Criando os valores SHAP para o modelo otimizado
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
shap_values = explainer.shap_values(Z)

# Plotando os valores SHAP
shap.summary_plot(shap_values[1], Z, feature_names=Z.columns)

# %% 
"""
Verificamos se y_test contém valores NaN. 
Se sim, removemos as linhas correspondentes em X.
"""
if np.isnan(y_test).any():
    print("y_test contém NaN. Removendo linhas correspondentes em X_test.")
    mask = ~np.isnan(y_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

print("Valores únicos em y_test:", np.unique(y_test))
print("Existem NaN em y_test?", np.isnan(y_test).any())

# %%
"""
Divisão do conjunto de dados

Dividimos o conjunto de dados em conjuntos de treino e validação.
Isso é importante para validar o modelo antes de testar em dados desconhecidos.
"""
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
"""
Verificação das dimensões dos conjuntos
"""
print("Dimensões de X_train:", X_train.shape)
print("Dimensões de y_train:", y_train.shape)
print("Dimensões de X_test:", X_test.shape)
print("Dimensões de y_test:", y_test.shape)

# %%
"""
Cálculo do scale_pos_weight

Calculamos o scale_pos_weight para lidar com o desbalanceamento das classes.
Isso ajuda o modelo a penalizar mais os erros na classe minoritária.
"""
num_positives = np.sum(y_train == 1)
num_negatives = np.sum(y_train == 0)
scale_pos_weight = num_negatives / num_positives

# %%
"""
Instanciação do modelo

Instanciamos o XGBClassifier com o scale_pos_weight para melhorar a capacidade do 
modelo em prever a classe minoritária.
"""
model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", use_label_encoder=False)

# %% 
"""
Limpeza dos dados

Removemos a coluna SAFRA_REF dos conjuntos de dados, pois não é necessária.
"""
X_train.drop(columns=['SAFRA_REF'], inplace=True, errors='ignore')
X_val.drop(columns=['SAFRA_REF'], inplace=True, errors='ignore')
X_test.drop(columns=['SAFRA_REF'], inplace=True, errors='ignore')

# %%
"""
Preparo da coluna DATA_EMISSAO_DOCUMENTO

Convertendo a coluna DATA_EMISSAO_DOCUMENTO para o formato datetime.
Isso permitirá a extração de características temporais.
"""
X_train['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(X_train['DATA_EMISSAO_DOCUMENTO'], errors='coerce')

# %%
"""
Extração de características de data

Função para extrair ano, mês e dia da coluna DATA_EMISSAO_DOCUMENTO.
Essas informações podem ser úteis para o modelo.
"""
def extract_date_features(df):
    df['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(df['DATA_EMISSAO_DOCUMENTO'], errors='coerce')
    df['EMISSAO_ANO'] = df['DATA_EMISSAO_DOCUMENTO'].dt.year
    df['EMISSAO_MES'] = df['DATA_EMISSAO_DOCUMENTO'].dt.month
    df['EMISSAO_DIA'] = df['DATA_EMISSAO_DOCUMENTO'].dt.day
    df.drop(columns=['DATA_EMISSAO_DOCUMENTO'], inplace=True, errors='ignore')
    return df

# Aplicar a função aos conjuntos
X_train = extract_date_features(X_train)
X_val = extract_date_features(X_val)
X_test = extract_date_features(X_test)

# %%
"""
Definição do pipeline

Definimos um pipeline que irá processar os dados e aplicar o modelo.
Isso facilita a aplicação das transformações e predições.
"""
column_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['VALOR_A_PAGAR', 'TAXA']),
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('classifier', XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss'))
])

# %%
"""
Ajuste do modelo

Ajustamos o pipeline aos dados de treinamento.
Isso prepara o modelo para fazer previsões.
"""
pipeline.fit(X_train, y_train)

# %%
"""
Previsões com o modelo

Fazemos previsões usando o conjunto de teste.
"""
y_pred_pipeline = pipeline.predict(X_test)

# %%
"""
Avaliação do modelo

Avaliamos o modelo utilizando a matriz de confusão e o relatório de classificação.
Isso nos ajuda a entender o desempenho do modelo.
"""
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_pipeline))

print("\nRelatório de Classificação do pipeline:")
print(classification_report(y_test, y_pred_pipeline, zero_division=0))

# %%
"""
Obter as probabilidades de inadimplência e criar um DataFrame final para o modelo XGBClassifier.

Aqui, armazenamos as previsões de inadimplência em um DataFrame.
"""
X_teste = base_pagamentos_teste.drop(['ID_CLIENTE'], axis=1)

# Converter DATA_EMISSAO_DOCUMENTO para datetime, se ainda não feito
X_teste['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(X_teste['DATA_EMISSAO_DOCUMENTO'], errors='coerce')

# Função para extrair ano, mês e dia
def extract_date_features(df):
    df['EMISSAO_ANO'] = df['DATA_EMISSAO_DOCUMENTO'].dt.year
    df['EMISSAO_MES'] = df['DATA_EMISSAO_DOCUMENTO'].dt.month
    df['EMISSAO_DIA'] = df['DATA_EMISSAO_DOCUMENTO'].dt.day
    df.drop(columns=['DATA_EMISSAO_DOCUMENTO'], inplace=True, errors='ignore')
    return df

# Aplicar a função a X_teste
X_teste = extract_date_features(X_teste)

# Garantir que as colunas em X_teste correspondam a X_train
X_teste = X_teste[X_train.columns]

# Agora você pode continuar com as previsões
resultado_probabilidade_xgb = model.predict_proba(X_teste)[:, 1]

resultado_final_xgb = pd.DataFrame({
    'ID_CLIENTE': base_pagamentos_teste['ID_CLIENTE'],
    'INADIMPLENTE': resultado_probabilidade_xgb
})

# %%
"""
Classificação de risco para o modelo XGBClassifier.

Definimos uma função para classificar o risco com base nas probabilidades de inadimplência.
"""
def classificar_risco(probabilidade):
    if probabilidade < 0.3:
        return 'Baixo Risco'
    elif 0.3 <= probabilidade <= 0.5:
        return 'Médio Risco'
    else:
        return 'Alto Risco'

resultado_final_xgb['Classificacao_Risco'] = resultado_final_xgb['INADIMPLENTE'].apply(classificar_risco)

# %%
"""
Salvando os resultados com a classificação de risco para o modelo XGBClassifier.

Por fim, salvamos o resultado em um arquivo CSV, além de criar a tabela geral que contém 
todas as informações do modelo.
"""
resultado_final_xgb.to_csv('resultado_inadimplencia_xgb.csv', index=False)

# Criando a tabela geral com a classificação de risco
geral_xgb = base_pagamentos_teste.copy()  # Cria uma cópia para não modificar o original
geral_xgb['Classificacao_Risco'] = resultado_final_xgb['Classificacao_Risco']  # Adicionando a classificação de risco
geral_xgb.to_csv('tabela_geral_xgb.csv', index=False)

# Exibir os resultados
print(resultado_final_xgb.head())

# %%
"""
Avaliação do resultado final do modelo XGBClassifier.

Aqui, avaliamos o desempenho do modelo com base nas previsões armazenadas em resultado_final_xgb.
"""
if np.isnan(y_test).any():
    print("y_test contém NaN. Removendo linhas correspondentes em X_test.")
    mask = ~np.isnan(y_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

# Fazer previsões com o pipeline ajustado
y_pred_pipeline = pipeline.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação do pipeline:")
print(classification_report(y_test, y_pred_pipeline, zero_division=0))

# Calcular e exibir a acurácia
accuracy = accuracy_score(y_test, y_pred_pipeline)
print(f"Acurácia: {accuracy:.4f}")


# %%
"""
Obtendo as probabilidades de inadimplência e criando um DataFrame final.

Aqui, armazenamos as previsões de inadimplência em um DataFrame.
"""
X_teste = base_pagamentos_teste.drop(['ID_CLIENTE'], axis=1)

resultado_probabilidade = pipeline.predict_proba(X_teste)[:, 1]

resultado_final = pd.DataFrame({
    'ID_CLIENTE': base_pagamentos_teste['ID_CLIENTE'],
    'INADIMPLENTE': resultado_probabilidade
})

# %%
"""
Classificação de risco.

Definimos uma função para classificar o risco com base nas probabilidades de inadimplência.
"""
def classificar_risco(probabilidade):
    if probabilidade < 0.3:
        return 'Baixo Risco'
    elif 0.3 <= probabilidade <= 0.5:
        return 'Médio Risco'
    else:
        return 'Alto Risco'

resultado_final['Classificacao_Risco'] = resultado_final['INADIMPLENTE'].apply(classificar_risco)

# %%
"""
Salvando os resultados com a classificação de risco.

Por fim, salvamos o resultado em um arquivo CSV, além de criar a tabela geral que contém 
todas as informações do modelo.
"""
resultado_final.to_csv('resultado_inadimplencia.csv', index=False)

# Criando a tabela geral com a classificação de risco
geral = base_pagamentos_teste.copy()  # Cria uma cópia para não modificar o original
geral['Classificacao_Risco'] = geral['INADIMPLENTES'].apply(classificar_risco)  # Adicionando a classificação de risco
geral.to_csv('tabela_geral.csv', index=False)

print(resultado_final.head())

