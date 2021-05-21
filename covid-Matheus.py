
#manipulação
import pandas as pd
import numpy as np

#visualização
import matplotlib.pyplot as plt
import seaborn as sns


#machine learning
#pip instal --user pycaret #nao tem permisao de adm
from pycaret.utils import enable_colab
enable_colab()

def diferenca_colum_df(list1,list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    print("A quantidade de colunas diferentes: ",len(c-d))
    return list(c - d) 

#Leitura
dados = pd.read_excel("https://github.com/alura-cursos/covid-19-clinical/blob/main/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx?raw=true")
dados.head() #visualizando as 5 primeiras linhas do dataset
linhas = dados.shape[0] #quantidade de linhas do dataset
colunas = dados.shape[1] #quantidade de colunas do dataset
print(f'O conjunto de dados extraído do Sírio Libanês contém {linhas} linhas e {colunas} colunas,')

#1.2
qt_pacientes = len(dados[dados['WINDOW'] == "0-2"])
print(f'O conjunto de dados contém {qt_pacientes} pacientes únicos.')


#1.4
dados[dados['WINDOW'] =='0-2'].isna().sum()


#1.5
def preenche_tabela(dados): #criando e definindo uma funçã
    '''
    O que esta função está realizando? De modo geral, procura  utilizar o groupby a partir do PATIENT_VISIT_IDENTIFIER, 
    Assim, para preencher os valores usando o bfill e o ffill
   '''
    features_continuas_colunas = dados.iloc[:, 13:-2].columns #determinando as colunas que contém valores contínuos 
    features_continuas = dados.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[features_continuas_colunas].fillna(method='bfill').fillna(method='ffill') #utilizando os métodos bfill e ffill
    features_categoricas = dados.iloc[:, :13] #determinando as colunas que são fatores
    saida = dados.iloc[:, -2:] 
    dados_finais = pd.concat([features_categoricas, features_continuas, saida], ignore_index=True,axis=1) #concatenando os dados a partir das colunas
    dados_finais.columns = dados.columns
    return dados_finais

dados_limpos = preenche_tabela(dados)
dados_limpos.head()


#1.6
dados_limpos[dados_limpos['WINDOW'] =='0-2'].isna().sum()
dados_limpos.loc[dados_limpos['DISEASE GROUPING 1'].isna()]


#1.7
dados_limpos = dados_limpos.dropna() #dropando as linhas/colunas que ainda possuem valores NaN.
dados_limpos[dados_limpos['WINDOW'] =='0-2'].isna().sum()


#1.8
pacientes_remover = dados_limpos.query("WINDOW=='0-2' and ICU==1")['PATIENT_VISIT_IDENTIFIER'].values
dados_limpos = dados_limpos.query("PATIENT_VISIT_IDENTIFIER not in @pacientes_remover")

linhas, colunas = dados_limpos.shape #extraindo o número de linhas e colunas do dataset atual
print(f'O dataset possui {linhas} linhas com {colunas} features.')



#1.9
def prepare_window(rows): #definindo e criando a função
    '''
    O que esta função está realizando? A mesma tenta adequar os dados de acordo com as diretrizes do Sírio-Libanês
    Onde visa extrair a informação de que o paciente foi ou não para UTI, independente da janela de tempo,
    assim, concatenando essa informação através do .loc na janela 0-2.
   '''
    if(np.any(rows["ICU"])):
        rows.loc[rows["WINDOW"]=="0-2", "ICU"] = 1
    return rows.loc[rows["WINDOW"] == "0-2"]


#mostra a idade dos pacientes da janela
alo = dados_limpos.copy()
dados_limpos = dados_limpos.groupby("PATIENT_VISIT_IDENTIFIER").apply(prepare_window).set_index('PATIENT_VISIT_IDENTIFIER').reset_index()

resposta = {'10th' : 0, '20th' : 1,'30th' : 2, '40th' : 3, '50th' : 4, '60th' : 5, '70th' : 6, '80th' : 7, '90th' : 8, 'Above 90th': 9}
dados_limpos.AGE_PERCENTIL = dados_limpos.AGE_PERCENTIL.replace(resposta)

linhas, colunas = dados_limpos.shape #extraindo o número de linhas e colunas do dataset atual
print(f'O dataset possui {linhas} linhas com {colunas} features.')

#352,231

#1.10
def create_histogram(data, title, bins = 5, xtitle = ' '):
  '''
  O que esta função faz? Extrai a média, a mediana e a moda dos dados, plota as mesmas na legenda,
  cria um histograma utilizando o seaborn
  '''
#  plt.figure(figsize=(10, 8))
  sns.set_theme(style="whitegrid")
  media = plt.axvline(x = data.mean(), c = "#05BA7F", linewidth = 4, linestyle = '--')
  mediana = plt.axvline(x = data.median(), c = "#A558C4", linewidth = 4, linestyle = ':')
  moda = plt.axvline(x = data.mode()[0], c = "#FF5733", linewidth = 4, linestyle = '-.')
  plt.legend([media, mediana, moda], ['Média', 'Mediana', 'Moda'])
  plt.xlabel(xtitle)
  sns.histplot(data, bins = bins).set_title(title, fontsize = 16);

plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
create_histogram(dados_limpos['ALBUMIN_MAX'], "Distribuição da coluna ALBUMIN_MAX")
plt.subplot(2,2,2)
create_histogram(dados_limpos['ALBUMIN_MEAN'], "Distribuição da coluna ALBUMIN_MEAN")
plt.subplot(2,2,3)
create_histogram(dados_limpos['ALBUMIN_MIN'], "Distribuição da coluna ALBUMIN_MIN")
plt.subplot(2,2,4)
create_histogram(dados_limpos['ALBUMIN_MEDIAN'], "Distribuição da coluna ALBUMIN_MEDIAN")


#1.11
nomenclaturas_removidas = ['MIN', 'MAX', 'MEDIAN', 'DIFF']
colunas_1_mudanca = dados_limpos.columns

lista = []
for nomenclatura in nomenclaturas_removidas:
  for column in dados_limpos.columns:
    if column.endswith(nomenclatura):
      lista.append(column)

dados_limpos_features = dados_limpos.drop(lista,axis=1)
colunas_2_mudanca = dados_limpos_features.columns
diferenca_colum_df(colunas_1_mudanca, dados_limpos_features.columns)
linhas, colunas = dados_limpos_features.shape
print(f'Os dados limpos no momentos, após a remoção das colunas indesejadas, contém {linhas} pacientes e {colunas} colunas.')


#1.12
def makebio_df(df:pd.DataFrame):
    
    df["BLOODPRESSURE_ARTERIAL_MEAN"] = (df['BLOODPRESSURE_SISTOLIC_MEAN'] + 2*df['BLOODPRESSURE_DIASTOLIC_MEAN'])/3
 
    # df["NEUTROPHILES/LINFOCITOS"] = df['NEUTROPHILES_MEAN']/df['LINFOCITOS_MEAN'] pode causar uma possível correlação
    
    df["GASO"] = df.groupby("PATIENT_VISIT_IDENTIFIER").P02_ARTERIAL_MEAN.apply(lambda x: x.fillna(method='ffill'))
    df["GASO"] = (~df["GASO"].isna()).astype(int)
    
    return df[["ICU","WINDOW",
               "PATIENT_VISIT_IDENTIFIER",
               "AGE_ABOVE65", 
               "GENDER", 
               "BLOODPRESSURE_ARTERIAL_MEAN", 
               "RESPIRATORY_RATE_MAX", 
               "HTN", 
               'DISEASE GROUPING 1',
               'DISEASE GROUPING 2',
               'DISEASE GROUPING 3',
               'DISEASE GROUPING 4',
               'DISEASE GROUPING 5',
               'DISEASE GROUPING 6',
               "GASO",
               "OXYGEN_SATURATION_MIN",
               "HEART_RATE_MAX",
               "PCR_MEAN",
               "CREATININ_MEAN"]]

features_sirio = dados_limpos.copy()
features_sirio = makebio_df(features_sirio)
features_sirio.head()

linhas, colunas = features_sirio.shape
print(f'Os dados com as colunas do Sírio-Libanês contém {linhas} pacientes e {colunas} colunas.')



#2.4
"""
quase metade dos dados é composto de paciente com mais de 65 anos.
"""
pd.DataFrame(dados_limpos_features['AGE_ABOVE65'].value_counts(normalize = True)).plot.barh()
plt.title("Análise da faixa etária dos pacientes")
plt.ylabel((' - de 65 anos', ' + de 65 anos'));



#2.5
#plot_correlation(dados_limpos_features, "LEUKOCYTES_MEAN", "NEUTROPHILES_MEAN") #Plotando a correlação entre as variáveis

# correlação entre as variáveis selecionadas a partir de um heatmap do seaborn
heatmap = sns.heatmap(dados_limpos_features[['LEUKOCYTES_MEAN', 'NEUTROPHILES_MEAN', 'HEMOGLOBIN_MEAN', 'HEMATOCRITE_MEAN']].corr(), cmap='Blues', fmt = '.2f')
plt.title('Correlação entre as variáveis');

plt.figure(figsize=(10,8))
dados_limpos_features[['LEUKOCYTES_MEAN', 'NEUTROPHILES_MEAN', 'HEMOGLOBIN_MEAN', 'HEMATOCRITE_MEAN']].boxplot().set_title("Distribuição das variáveis analisadas");


#2.8

dados_concatenados = pd.merge(dados_limpos_features, features_sirio, how = 'left', left_on=['PATIENT_VISIT_IDENTIFIER'], right_on = ['PATIENT_VISIT_IDENTIFIER'])
diferenca_colum_df(colunas_2_mudanca, dados_concatenados.columns)

colunas_repetidas = []

for columns in dados_concatenados.columns:
  if columns.endswith('y'):
    colunas_repetidas.append(columns)

colunas_3_mudanca = dados_concatenados.columns
dados_concatenados = dados_concatenados.drop(colunas_repetidas,axis=1)
dados_concatenados.columns = dados_concatenados.columns.str.replace("_x", "")
diferenca_colum_df(colunas_3_mudanca, dados_concatenados.columns)

linhas, colunas = dados_concatenados.shape
print(f'O dataset concatenado possui {linhas} pacientes e {colunas} features. O dataset original possuía {dados.shape[1]} colunas. Redução de {dados.shape[1] - colunas} features.')


#2.9
dados_concatenados = dados_concatenados.drop(['WINDOW', 'GASO'], axis=1)
colunas = dados_concatenados.shape[1]


print(f'O dataset possui {colunas} colunas agora.')


#2.10
colunas_4_mudanca = dados_concatenados.columns
dados_concatenados = dados_concatenados.drop(['HEMATOCRITE_MEAN', 'LEUKOCYTES_MEAN'], axis = 1).set_index('PATIENT_VISIT_IDENTIFIER')
diferenca_colum_df(colunas_4_mudanca, dados_concatenados.columns)

#2.11
plt.figure(figsize=(20,12))
x = np.triu(np.ones_like(dados_concatenados.corr(), dtype=np.bool)) #matriz para evitar a "diagonal principal à direita"
heatmap = sns.heatmap(dados_concatenados.corr(), mask = x, cmap='BuPu_r', fmt = '.2f')
plt.title('Correlação entre as variáveis', fontsize = 18);


#2.12
"""
dados_concatenados_090 = dados_concatenados.copy()
alta_corr = 0.90 #determinando a correlação
matrix_corr = dados_concatenados_090.corr().abs()
matrix_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
excluir = [coluna for coluna in matrix_upper.columns if any(matrix_upper[coluna] > alta_corr)] #determinando as colunas que devem ser excluídas

dados_concatenados_090 = dados_concatenados_090.drop(excluir, axis=1)
linhas, colunas = dados_concatenados_090.shape #extraindo o número de linhas e colunas do dataset atual
print(f'O dataset possui {linhas} linhas com {colunas} features.')
"""

alta_corr = 0.81 #determinando a correlação
matrix_corr = dados_concatenados.corr().abs()
matrix_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
excluir = [coluna for coluna in matrix_upper.columns if any(matrix_upper[coluna] > alta_corr)] #determinando as colunas que devem ser excluídas

dados_concatenados = dados_concatenados.drop(excluir, axis=1)
linhas, colunas = dados_concatenados.shape #extraindo o número de linhas e colunas do dataset atual
print(f'O dataset possui {linhas} linhas com {colunas} features.')

colunas_finais_matheus = dados_concatenados.columns
colunas_finais_matheus = colunas_finais_matheus.sort_values()


#6.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

rf_model_grid_search_best_params = RandomForestClassifier(random_state = 0, n_estimators = 1000, **gs_grid.best_params_).fit(X_train, y_train)

print("RandomForest Otimizado utilizando GridSearch\n\n")
avaliacao(rf_model_grid_search_best_params, X_test, y_test, X_train, y_train, train_aval = True, plot=True)



#6.4
X = dados_concatenados.drop('ICU', axis = 1)
y = dados_concatenados['ICU']

best_params = {'bootstrap': True,
 'max_depth': 20,
 'max_features': 11,
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 1000,
 'random_state': 0}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

rf_gs_corte_model = RandomForestClassifier(**best_params).fit(X_train, y_train)
avaliacao(rf_gs_corte_model, X_test, y_test, X_train, y_train, train_aval = True, plot=True)














































































































































































































































