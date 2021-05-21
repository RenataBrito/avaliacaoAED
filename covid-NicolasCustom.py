# NumPy, Pandas & Visualização de Dados
import numpy as np
from numpy import mean
from scipy import stats
from numpy import absolute
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Scikit-Learn Metrics, Split and Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, RandomizedSearchCV, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, plot_confusion_matrix, f1_score, precision_score, accuracy_score, roc_curve, auc

# Scikit-Learn & XGBoost Algorithms 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Funções

#1
def colunas_correlacionadas(dataset, threshold = 0.95):
    
  '''   
  Retorna um dataframe com 3 colunas, sendo 
  duas delas o nome de colunas do dataframe de entrada e a 
  terceira o nível de correlação entre elas. Só serão retornadas
  colunas com correlação pearson com o valor mínimo definido no threshold
  
  Crédito: Matt Harrison @Machine Learning Pocket Reference
  '''

  df = dataset[dataset.describe().columns]  
  return (
      df.corr().pipe(lambda df1: pd.DataFrame(np.tril(df1,k=-1),
                                              columns = df.columns,
                                              index = df.columns,
                                              )
      )
      .stack()
      .rename("pearson")
      .pipe(
          lambda s: s[s.abs() >threshold].reset_index()
      )
      .query("level_0 not in level_1")
    )

#2
def makebio_df(df_in):
    
    '''
    Retorna um dataframe com as features consideradas pelos autores
    do dataset como biologicamente importantes
    '''
    
    df = df_in.copy()

    df["BLOODPRESSURE_ARTERIAL_MEAN"] = (df.loc[:,'BLOODPRESSURE_SISTOLIC_MEAN'] + 2*df.loc[:,'BLOODPRESSURE_DIASTOLIC_MEAN'])/3
 
    df["NEUTROPHILES/LINFOCITOS"] = df.loc[:,'NEUTROPHILES_MEAN']/df.loc[:,'LINFOCITOS_MEAN']

    df["GASO"] = df.groupby("PATIENT_VISIT_IDENTIFIER").P02_ARTERIAL_MEAN.apply(lambda x: x.fillna(method='ffill'))
    df["GASO"] = (~df.loc[:,"GASO"].isna()).astype(int)

    return df[["ICU_ANYTIME",
               "AGE_ABOVE65", 
               "GENDER", 
               "BLOODPRESSURE_ARTERIAL_MEAN", 
               "RESPIRATORY_RATE_MAX", 
               "HTN",
               'P02_ARTERIAL_MEAN',
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

#3
def avaliacao(model, X_test, y_test, X_train = None, y_train = None, train_aval = False, plot = False, cria_df = False):
    
    '''
    Calcula algumas medidas de score - Acurácia, Precisão, Recall e ROC(AUC).
    Se plot for definido como True, plota esses scores e uma matriz de 
    confusão para avaliação do modelo. Se cria_df for definido como True, retorna
    um dataframe com os resultados (para a função exec_modelagem)
    '''
    if train_aval:
        # TRAIN SCORES

        tn, fp, fn, tp = confusion_matrix(y_train, model.predict(X_train)).ravel()

        # Calcula os scores
        roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
        recall = recall_score(y_train, model.predict(X_train))
        especificidade = tn/(tn+fn)
        f_score = f1_score(y_train, model.predict(X_train))
        precisao = precision_score(y_train, model.predict(X_train))
        acuracia = accuracy_score(y_train, model.predict(X_train))

        # Printa os scores
        if plot:
            print("TRAIN SET SCORES")
            print(f"ROC (AUC): {roc_auc}")
            print(f"Sensibilidade - Recall: {recall}")
            print(f"Especificidade: {especificidade}")
            print(f"F1-score: {f_score}")
            print(f"Precisão: {precisao}")
            print(f"Acurácia: {acuracia}")
            print("\n")
            print("================================")
    
    
    # TEST SCORES
    
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    
    # Calcula os scores
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    recall = recall_score(y_test, model.predict(X_test))
    especificidade = tn/(tn+fn)
    f_score = f1_score(y_test, model.predict(X_test))
    precisao = precision_score(y_test, model.predict(X_test))
    acuracia = accuracy_score(y_test, model.predict(X_test))
    
    # Printa os scores e plota a matriz
    if plot:
        print("TEST SET SCORES")
        print(f"ROC (AUC): {roc_auc}")
        print(f"Sensibilidade - Recall: {recall}")
        print(f"Especificidade: {especificidade}")
        print(f"F1-score: {f_score}")
        print(f"Precisão: {precisao}")
        print(f"Acurácia: {acuracia}")
        
        plot_confusion_matrix(estimator = model, X = X_test, y_true = y_test)
    
    # Retorna uma row com os scores
    if cria_df:
        d = {
            'ROC (AUC)': [roc_auc],  
            'Sensibilidade - Recall': [recall],
            'Especificidade': [especificidade],
            'F1-score': [f_score],
            'Precisão': [precisao],
            'Acurácia': [acuracia]
        }
        return(pd.DataFrame(d))
    
    else:
        return None
    

#4
def exec_modelagem(df, model, n_iter = 50, resumo = False, **params):
    
    '''
    Para evitar a aleatoriedade dos modelos (Aula 3 - @Bootcamp), ele será rodado
    n_iter vezes utilizando o mesmo stratify (mas com random_state variável).
    Retorna um dataframe com os resultados - usando a funcao avaliacao() - de cada
    iteração. Se resumo for definido como True, printa a média dos scores.
    '''
    # Cria o df para armazenamento dos scores
    col = {
            'ROC (AUC)': [],  
            'Sensibilidade - Recall': [],
            'Especificidade': [],
            'F1-score': [],
            'Precisão': [],
            'Acurácia': []
        }
    df_scores = pd.DataFrame(col)
    
    # Roda o modelo n_iter vezes, avalia utilizando a função `avalicao()` e armazena os resultados 
    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(df.drop('ICU', axis = 1), df['ICU'], test_size=0.25, stratify=df['ICU'], random_state=i)
        
        if model == LogisticRegression:
            model_ = model(max_iter=2000, random_state = i).fit(X_train, y_train)
            
        elif model == XGBClassifier:
            model_ = model(**params, eval_metric = 'error', use_label_encoder=False, random_state=i).fit(X_train, y_train)
            
        else:
            model_ = model(**params, random_state = i).fit(X_train, y_train)

            
        current_result = avaliacao(model_, X_test, y_test, cria_df = True, train_aval = False)
        df_scores = df_scores.append(current_result)
        
    if resumo:
        return df_scores.median()
    
    else:
        return df_scores.reset_index()
    
#5
def kfold_cross_validation(X, y, k=10):
    
    # Cria o df para armazenamento dos scores
    col = {
            'model':[],
            'ROC (AUC) MEDIANA': [],
            'ROC (AUC) STD': [],
            'Sensibilidade - Recall MEDIANA': [],
            'Sensibilidade - Recall STD': [],
            'F1-score MEDIANA': [],
            'F1-score STD': [],
            }
    df_scores = pd.DataFrame(col)

    for model in [DecisionTreeClassifier, KNeighborsClassifier, GaussianNB, SVC, RandomForestClassifier, XGBClassifier]:


        # Instanciar o 10-fold usando cross_validate c/ 3 scores

        # Evitar warning no XGBC
        if model == XGBClassifier:
            cls = model(eval_metric = 'error', use_label_encoder=False, random_state=42)
            kfold = KFold(n_splits=10, random_state=42, shuffle=True)
            scores = cross_validate(cls, X, y, scoring=['roc_auc','recall','f1'], cv=kfold)

        elif model in [KNeighborsClassifier, GaussianNB]:
            cls = model()
            kfold = KFold(n_splits=10, random_state=42, shuffle=True)
            scores = cross_validate(cls, X, y, scoring=['roc_auc','recall','f1'], cv=kfold)
            
        else:
            cls = model(random_state=42)
            kfold = KFold(n_splits=10, random_state=42, shuffle=True)
            scores = cross_validate(cls, X, y, scoring=['roc_auc','recall','f1'], cv=kfold)
            
        # Armazenar os scores
        scores = {
                'model': f'{model.__name__:22}',
                'ROC (AUC) MEDIANA': f"{np.median(scores['test_roc_auc']):.3f}",
                'ROC (AUC) STD': f"{scores['test_roc_auc'].std():.2f}",
                'Sensibilidade - Recall MEDIANA': f"{np.median(scores['test_recall']):.3f}",
                'Sensibilidade - Recall STD': f"{scores['test_recall'].std():.2f}",
                'F1-score MEDIANA': f"{np.median(scores['test_f1']):.3f}",
                'F1-score STD': f"{scores['test_f1'].std():.2f}",
                }
        df_scores = df_scores.append(scores, ignore_index=True)


    return df_scores

#6
def plot_grid_search_params(grid_search_model):
    '''
    Essa função usa o plotly para plotar um mapa 2D dos resultados da
    autoparametrização do GridSearchCV() - dessa forma facilitando a 
    visualização de outros possíveis parâmetros bons
    
    Credit: Data Professor at 
    https://github.com/dataprofessor/code/blob/master/python/hyperparameter_tuning.ipynb
    '''
    
    # Exportar os resultados e dropar a coluna Random State
    grid_results = pd.concat([pd.DataFrame(grid_search_model.cv_results_["params"]),
                              pd.DataFrame(grid_search_model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    
    # Segmentar os dados em grupos dos dois hyperparametros
    # Após, pivorar os dados feature_1 x feature_2
    grid_contour = grid_results.groupby(['max_features','max_depth']).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'max_depth', 'Accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'max_depth')
    
    # Separar os dados pivotados para o plot
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # Criação do Plot
    layout = go.Layout(
                xaxis=go.layout.XAxis(
                  title=go.layout.xaxis.Title(
                  text='max_depth')
                 ),
                 yaxis=go.layout.YAxis(
                  title=go.layout.yaxis.Title(
                  text='max_features') 
                ) )

    fig = go.Figure(data = [go.Contour(z=z, x=x, y=y)], layout=layout )

    fig.update_layout(title='Hyperparameter tuning', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()
    
    
#7
def plot_feature_importance(model):
    ax, fig = plt.subplots(figsize=(25,18))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


#10
def teste_de_normalidade(df):
    column_names = []
    column_names.clear()
    for i in df.columns:
        if any(x in i for x in ['_MEAN']):
            column_names.append(i)
    
    df_testeNormalidade = X.loc[:, column_names]
    p_values = []
    
    for i in range(len(column_names)):
        shapiro_test = stats.shapiro(df_testeNormalidade.iloc[:,i:i+1])
        b = shapiro_test.pvalue
        p_values.append(b)
    
    
    return p_values



#Leitura dos dados
raw_df = pd.read_excel("https://github.com/alura-cursos/covid-19-clinical/blob/main/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx?raw=true")

#Analise

#2.3
# Identificando as colunas que possuem NaNs
colunas_com_nan = raw_df.isna().sum()[raw_df.isna().sum() != 0].index.tolist()

# Utilizando o fillna para preencher os valores NaN nas colunas que os possuem; 
df_sem_nan = raw_df.copy()

df_sem_nan[colunas_com_nan] = df_sem_nan.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False)[colunas_com_nan].fillna(method='bfill')
df_sem_nan[colunas_com_nan] = df_sem_nan.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False)[colunas_com_nan].fillna(method='ffill')


#linha que ainda tinha NaN e dropa
df_sem_nan[df_sem_nan.isnull().any(axis=1)]
df_sem_nan.dropna(inplace=True)


#2.8
colunas_de_interesse = ['PATIENT_VISIT_IDENTIFIER', 'AGE_ABOVE65','AGE_PERCENTIL','GENDER','IMMUNOCOMPROMISED','RESPIRATORY_RATE_MEDIAN','DISEASE GROUPING 1',
                        'DISEASE GROUPING 2','DISEASE GROUPING 3','DISEASE GROUPING 4','DISEASE GROUPING 5','DISEASE GROUPING 6']


# Para realizar os comparativos, utilizarei os primeiros dados após a entrada da UTI para os pacientes ICU == 1
pacientes_uti = df_sem_nan[df_sem_nan['ICU'] == 1][colunas_de_interesse]\
                .drop_duplicates(subset = 'PATIENT_VISIT_IDENTIFIER', keep = 'first')

lista_id_pacientes_uti = pacientes_uti['PATIENT_VISIT_IDENTIFIER'].tolist()


# E de entrada no hospital para os pacientes ICU == 0
pacientes_nao_uti = df_sem_nan.query("PATIENT_VISIT_IDENTIFIER not in @lista_id_pacientes_uti")[colunas_de_interesse]\
                    .drop_duplicates(subset = 'PATIENT_VISIT_IDENTIFIER', keep = 'first')



#2.11
"""
AGE_ABOVE65 - Pessoas acima de 65 anos são mais comuns dentro da UTI, como era esperado pelos resultados do countplot da coluna 'AGE_PERCENTIL'.

GENDER - Apesar de pessoas do gênero '0' serem mais comuns em ambos subsets, elas são ainda mais comuns dentro da UTI.

IMMUNOCOMPROMISED e DISEASE GROUPING i - As distribuições são parecidas entre os subsets, mas é válido a observação de que pessoas positivas (com a exceção da coluna DISEASE GROUPING 6) são mais comuns no set de pessoas que foram encaminhadas para a UTI.

"""
colunas_binarias = [
 'AGE_ABOVE65',
 'GENDER',
 'IMMUNOCOMPROMISED',
 'DISEASE GROUPING 1',
 'DISEASE GROUPING 2',
 'DISEASE GROUPING 3',
 'DISEASE GROUPING 4',
 'DISEASE GROUPING 5',
 'DISEASE GROUPING 6']


#2.12
"""
Para confirmar a existência dessas váriaveis altamente correlacionadas, 
utilizarei a função do livro Machine Learning Pocket Reference, 
escrito por Matt Harrison, 
que mostra as colunas correlacionadas dado algum limite mínimo de correlação pearson.

"""
pd.set_option("display.max_rows", None)
colunas_correlacionadas(df_sem_nan)


#df_sem_nan
#3.1
#231 colunas
pacientes_a_remover = df_sem_nan.query("WINDOW == '0-2' & ICU == 1")["PATIENT_VISIT_IDENTIFIER"]
#tirou alguns pacientes
df_processado = df_sem_nan.query("PATIENT_VISIT_IDENTIFIER not in @pacientes_a_remover")


#3.2.1
# Pacientes que foram pra UTI depois da janela de 2 hrs
paciente_uti = df_processado.groupby('PATIENT_VISIT_IDENTIFIER').agg({'ICU': max}).rename(columns={'ICU': 'ICU_ANYTIME'})

    
# Adicionando a coluna criada acima no df
#232 colunas
df_processado = df_processado.merge(paciente_uti, on=['PATIENT_VISIT_IDENTIFIER'], how = 'right')

#3.2.2
# Mantendo apenas os dados de pacientes entre [0-2]
#tirou paciente de outras janelas
df_processado = df_processado[df_processado['WINDOW'] == '0-2']

#3.3
df_bio =  makebio_df(df_processado)


#3.2.3
# Transformando GENDER e AGE_ABOVE65 em strings para serem processados pelo get_dummies()
df_processado['GENDER'] = df_processado['GENDER'].astype(str)
df_processado['AGE_ABOVE65'] = df_processado['AGE_ABOVE65'].astype(str)
df_processado = pd.get_dummies(df_processado, columns=['GENDER','AGE_PERCENTIL','AGE_ABOVE65'], prefix=['GENDER','AGE_PERCENTIL','AGE_ABOVE65'])


#3.3.1
#so possui valor um
print(df_bio['GASO'].nunique())
df_bio.drop('GASO', axis = 1, inplace = True)


#3.4
# Apagando colunas ICU(antiga), WINDOW e PATIENT_VISIT_IDENTIFIER

df_processado.drop('ICU', axis=1, inplace=True)
df_processado.drop('WINDOW', axis=1, inplace=True)
df_processado.drop('PATIENT_VISIT_IDENTIFIER', axis=1, inplace=True)


#3.5
for df in [df_processado, df_bio]:
    df['ICU'] = df['ICU_ANYTIME']
    df.drop('ICU_ANYTIME', axis=1, inplace=True)
    
    
#3.6
df_modelo = df_processado.copy()

# Criar lista com colunas que possuem valor constante
colunas_const = list(df_modelo.columns[df_modelo.nunique() == 1])

# Dropar do dataset que vai ser usado na modelagem
for col in colunas_const:
    df_modelo.drop(col, axis=1, inplace=True, errors='ignore')


# Criarei um df_modelo_mediana onde utilizarei apenas a mediana das medições
df_modelo_mediana = df_modelo.copy()
#df_modelo_media = df_modelo.copy()


for col in df_modelo_mediana.columns:
    if any(x in col for x in ['_MAX', '_MIN', '_MEAN']):
            df_modelo_mediana.drop(col, axis=1, inplace=True, errors='ignore')


#for col in df_modelo_media.columns:
#    if any(x in col for x in ['_MAX', '_MIN', '_MEDIAN','_DIFF']):
#            df_modelo_media.drop(col, axis=1, inplace=True, errors='ignore')

#for i in df_modelo_media.columns:
#    print(i)

#column_names = []
#column_names.clear()
#for i in df_modelo_media.columns:
#    if any(x in i for x in ['_MEAN']):
#        column_names.append(i)
    
    #    df_testeNormalidade = df_modelo_media.loc[:, column_names]
    #    p_values = []
    
#for i in range(len(column_names)):
#    shapiro_test = stats.shapiro(df_testeNormalidade.iloc[:,i:i+1])
#    b = shapiro_test.pvalue
#    p_values.append(b)

#for i in p_values:
#    if i > 0.005:
#        print("usa a media")

# Remover as colunas com alta correlação
for col in colunas_correlacionadas(df_modelo, threshold = 0.90)["level_1"]:
    df_modelo.drop(col, axis=1, inplace=True, errors='ignore')



#colunas apagadas
#pd.set_option("display.max_rows", None)
#colunas_correlacionadas(df_modelo_mediana, threshold = 0.90)



#4.2 + mediana + diff (distribuicao dentro e fora da UTI)
variaveis_continuas = list(df_modelo_mediana.iloc[:, 9:-15].columns)
for col in variaveis_continuas:
    fig, axs = plt.subplots(2, sharex=True, sharey=True)

    sns.boxplot(data = df_modelo_mediana[df_modelo['ICU'] == 1], x = col, ax = axs[0], color = 'b');
    sns.boxplot(data = df_modelo_mediana[df_modelo['ICU'] == 0], x = col, ax = axs[1], color = 'g');

    axs[0].set_title('Dentro da UTI')
    axs[0].set_xlabel('')
    axs[1].title.set_text('Fora da UTI')

    
    
#4.3
features_dist_identicas = ['BE_VENOUS_MEDIAN', 'CALCIUM_MEDIAN', 'HEMOGLOBIN_MEDIAN', 'P02_VENOUS_MEDIAN', 'SAT02_VENOUS_MEDIAN', 'HEART_RATE_MEDIAN']

features_proximas_c_outliers_expressivos = ['ALBUMIN_MEDIAN','BE_ARTERIAL_MEDIAN','BIC_ARTERIAL_MEDIAN','BIC_VENOUS_MEDIAN',
'BILLIRUBIN_MEDIAN','BLAST_MEDIAN','FFA_MEDIAN','GGT_MEDIAN','CREATININ_MEDIAN', 'P02_ARTERIAL_MEDIAN','PC02_ARTERIAL_MEDIAN',
'PH_ARTERIAL_MEDIAN','SAT02_ARTERIAL_MEDIAN','TGO_MEDIAN','TGP_MEDIAN','TTPA_MEDIAN','DIMER_MEDIAN','OXYGEN_SATURATION_MEDIAN',
'BLOODPRESSURE_DIASTOLIC_DIFF','BLOODPRESSURE_SISTOLIC_DIFF','HEART_RATE_DIFF','RESPIRATORY_RATE_DIFF','TEMPERATURE_DIFF',
'OXYGEN_SATURATION_DIFF','BLOODPRESSURE_DIASTOLIC_DIFF_REL','BLOODPRESSURE_SISTOLIC_DIFF_REL','HEART_RATE_DIFF_REL',
'RESPIRATORY_RATE_DIFF_REL','TEMPERATURE_DIFF_REL','OXYGEN_SATURATION_DIFF_REL']

featuers_dist_diferentes = ['GLUCOSE_MEDIAN','HEMATOCRITE_MEDIAN','INR_MEDIAN','LACTATE_MEDIAN',
'LEUKOCYTES_MEDIAN','LINFOCITOS_MEDIAN','NEUTROPHILES_MEDIAN','PC02_VENOUS_MEDIAN',
'PCR_MEDIAN','PH_VENOUS_MEDIAN','PLATELETS_MEDIAN','POTASSIUM_MEDIAN','SODIUM_MEDIAN','UREA_MEDIAN',
'BLOODPRESSURE_DIASTOLIC_MEDIAN','BLOODPRESSURE_SISTOLIC_MEDIAN','RESPIRATORY_RATE_MEDIAN','TEMPERATURE_MEDIAN']


#5.1

dfs = [('DF CORR CORTADA',df_modelo),('DF MEDIANA',df_modelo_mediana),('DF BIO',df_bio)]


#5.2
for df in dfs:
    X = df[1].drop('ICU', axis=1)
    y = df[1]['ICU']
    print(f"==== SCORE {df[0]} =====")
    if df[0] == "DF BIO":
        df_bio_cross_validation = kfold_cross_validation(X,y)
    elif df[0] == "DF MEDIANA": 
        df_MEDIANA_cross_validation = kfold_cross_validation(X,y)
    elif df[0] == "DF CORR CORTADA":
        df_CORRCORTADA_cross_validation = kfold_cross_validation(X,y)


#5.3
for df in dfs:
    print(f"==== SCORE {df[0]} RANDOM FOREST DEFAULT =====")
    display(exec_modelagem(df[1], RandomForestClassifier, n_iter = 50, resumo = True))



#corte
"""
identifiquei nas features contínuas alguns comportamentos diferentes 
quanto às distribuições ao comparar os pacientes dentro e fora da UTI. 
Antes de fazer o ajuste dos hiperparâmetros, irei testar se cortar algumas 
features do dataset 2 trará benefícios ao modelo.
"""

# Dataset 2 sem as features que possuíam distribuições iguais entre pacientes dentro e fora da UTI
df_modelo_mediana_corte = df_modelo_mediana.drop(features_dist_identicas, axis = 1)


#6.1 e 6.2
X = df_modelo_mediana_corte.drop('ICU', axis = 1)
y = df_modelo_mediana_corte['ICU']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)


param_grid = {'max_depth': [i for i in range(1,20)],
               'max_features':[i for i in range(1,12)]}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_model_GS = RandomForestClassifier(random_state = 42)

gs_grid = GridSearchCV(rf_model_GS, param_grid, scoring='f1', cv = 5, verbose=1, n_jobs = -1)
gs_grid.fit(X_train, y_train)


print(gs_grid.best_params_)



#6.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

rf_model_grid_search_best_params = RandomForestClassifier(random_state = 0, n_estimators = 1000, **gs_grid.best_params_).fit(X_train, y_train)

print("RandomForest Otimizado utilizando GridSearch\n\n")
avaliacao(rf_model_grid_search_best_params, X_test, y_test, X_train, y_train, train_aval = True, plot=True)



#6.4
X = df_modelo_mediana_corte.drop('ICU', axis = 1) 
y = df_modelo_mediana_corte['ICU']

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












