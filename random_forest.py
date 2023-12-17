#!/usr/bin/env python
# coding: utf-8

# # 1.Configuration et Import

# In[123]:


get_ipython().system('pip install imblearn')
get_ipython().system('pip install scikit-optimize')
get_ipython().system('pip install ipython-autotime')


# In[26]:


# imports

# pandas
import pandas as pd

# numpy
import numpy as np

# math
import math

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

# Visualisation des données
from matplotlib import pyplot as plt
from plotly import subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
from plotly.subplots import make_subplots

sns.set(rc={'figure.figsize':(11.7,8.27)})


# SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler 


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from collections import Counter

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

get_ipython().run_line_magic('load_ext', 'autotime')


# In[90]:


save_path = '/Users/sulifang/Desktop/fouille_slide/'


# In[77]:


# path = '/Users/sulifang/Desktop/M2/fouil-donnee/proejet/'
df = pd.read_csv('creditcard.csv')


# In[141]:


df.head()


# In[6]:


df.isnull().any()


# >Il n'y a pas de valeurs manquantes dans les données et aucun traitement des valeurs manquantes n'est nécessaire.

# # 2.Feature Engineering
# Le Feature Engineering est un processus qui consiste à transformer les données brutes en caractéristiques représentant plus précisément le problème sous-jacent au modèle prédictif. https://datascientest.com/feature-engineering
# 
# Comme les colonnes V1 à V28, les données sont transformés par PCA, on a pas besoin de faire la Feature Extraction. Par contre, les colonnes "Time" et "Amount" ont les types de données très différents par rapport les autres, il faut faire la **Feature scaling**. 

# ## 2.1 Transformer la colonne 'Time' en heure de la journée

# In[142]:


def convert_hour(x):
    if x > 23:
        return x-24
    else:
        return x


# In[143]:


def transformeTimeToHour(dataframe, convertFunc):
    dataframe['Hour'] = dataframe['Time'].apply(lambda x : divmod(x, 3600)[0])
    dataframe['hour_of_day'] = dataframe['Hour'].apply(lambda x : convertFunc(x))
    return dataframe


# In[144]:


df = transformeTimeToHour(df, convert_hour)


# In[10]:


df.describe()


# ## 2.2 L'analyse exploratoire des données (AED) : corrélation 

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
corr = df.corr()


# In[14]:


layout = go.Layout(autosize=False, width=750, height=750) 

fig = go.Figure(layout = layout)

fig.add_trace(go.Heatmap(
    z=corr.to_numpy().round(2),
    x=list(corr.index.values),
    y=list(corr.columns.values),
    colorscale="RdBu_r"
))
fig.show()


# ## 2.3 La corrélation par la classe
# La corrélation entre 2 variables de dataset par rapport à la classe. 
# 
# Nous recherchons la correlation entre les variables. Nous supposons que les relations suivent la loi normal.  
# 
# >Selon les figure ci-dessus, dans la situation de fraud, les corrélations entre certaines variables sont plus prononcées. La variation entre V1, V2, V3, V4, V5, V6, V7, V9, V10, V11, V12, V14, V16, V17, V18 et V19 montre un certain schéma dans l'échantillon d'écrémage de cartes de crédit.

# In[12]:


df_fraud = df.loc[df["Class"]==1]
df_nonfraud = df.loc[df["Class"]==0]

# regarder la correlation entre les variables
# Ici, on suppose que la relation est linéaire, la valeur est dans [-1,1]
# Si la valeur vaut 1, les variables sont en relation linéaire
corrs_nonfraud = df_nonfraud.loc[:, df.columns != "Class"].corr()
corrs_fraud = df_fraud.loc[:, df.columns != "Class"].corr()

# afficher seulement le triangle
mask_fraud = np.triu(np.ones_like(corrs_fraud, dtype=bool))
corrs_fraud = corrs_fraud.mask(mask_fraud)
mask_nonfraud = np.triu(np.ones_like(corrs_nonfraud, dtype=bool))
corrs_nonfraud = corrs_nonfraud.mask(mask_nonfraud)


# In[13]:


fig_nonfraud = go.Heatmap(
    z=corrs_nonfraud.to_numpy().round(2),
    x=list(corrs_nonfraud.index.values),
    y=list(corrs_nonfraud.columns.values),
    colorscale="RdBu_r"
)

fig_fraud = go.Heatmap(
    z=corrs_fraud.to_numpy().round(2),
    x=list(corrs_fraud.index.values),
    y=list(corrs_fraud.columns.values),
    colorscale="RdBu_r"
)

#regrouper les deux graphes
fig = subplots.make_subplots(rows = 1, cols = 2, shared_yaxes=True)
fig.append_trace(fig_nonfraud, 1, 1)
fig.append_trace(fig_fraud, 1, 2)
fig.update_xaxes(showgrid=False, row=1, col=1)
fig.update_xaxes(showgrid=False, row=1, col=2)
fig.update_yaxes(showgrid=False, row=1, col=1, autorange='reversed')
fig.update_yaxes(showgrid=False, row=1, col=2, autorange='reversed')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
fig.show()


# Ensuite, on peut vérifier s'il existe une colinéartié enntre les variable. 
# > Avec le résultat ci-dessus, on détecte une colinéarité entre V16 et V17 sur le dataset de fraude. On doit supprimer l'une des deux variables. 

# In[14]:


colinearites = (corrs_fraud > 0.95).any()
for i in range(1, 28):
  if colinearites['V'+str(i)] == True:
    print('V'+str(i), colinearites['V'+str(i)])


# ## 2.4 Le nombre de fraudes au cours du temps 
# On calcule le nombre de transactions au cours de temps. 
# > Avec le graphe ci-dessus, on regarde uniquement la distribution des fraudes par rapport l'heure de la journée. 
# 

# In[ ]:


df_fraud_temps = df_fraud
df_fraud_temps['count'] = df_fraud.groupby('hour_of_day')['hour_of_day'].transform('count')
graphe_temps = sns.catplot(data=df_fraud_temps, x="hour_of_day", y="count", kind="bar", height=8, aspect=2)


# Ensuite, en regardant le graphe ci-dessus, on compare la variable V17 en fraud et non fraud. 
# 
# La médian de fraud_V17 change tout le temps. Donc, on a une signal assez forte pour détecter le fraud.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=False)

graphe_fraud_v17 = sns.boxplot(df_fraud['hour_of_day'], df_fraud['V17'], ax=axes[0], showfliers = False).set_title('fraud_V17')
graphe_fraud_v17 = sns.boxplot(df_nonfraud['hour_of_day'], df_nonfraud['V17'], ax=axes[1], showfliers = False).set_title('Non fraud_V17')

# fig.savefig(save_path + 'v17_fraud_nonfraud_boxplot.jpg')


# ## 2.5 Le médian de chaque variable

# Pour bien choisir les variable utilisées dans notre modèle, on regarde le médian de chaque variable en fonciton de la calsse. 

# In[ ]:


fig, axes = plt.subplots(7, 4, figsize=(36, 42), sharey=False)

k_tmp=0
for i in range(0,7):
  for j in range(0, 4):
    if k_tmp <= 27:
      k_tmp=k_tmp+1
    sns.boxplot(df['Class'], df['V'+str(k_tmp)], ax=axes[i, j], showfliers = False)

# fig.savefig(save_path + 'median_class_variable2.svg')


# Avec les graphes au dessus, on regarde les médians de chauqe variable par rapport à sa classe, si les médians sont proches, ce sera plus difficile à détecter les choses intérassants. 
# 
# > **Bien : V1, V2, V3, V4, V6, V7, V8, V9, V10, V11, V12, V14, V17, V18, V19, V20, V21, V27**
# 
# > **Moins bien : V5, V13, V15, V16, V22, V23, V24, V25, V26, V28** 
# 
# 

# ## 2.6 Distribution des données
# Nous regardons à nouveau la distribution des données. Nous préférons sélectionner des variables qui ont des distributions significativement différentes sous différentes classes. 
# 
# On combinane les résultats de boxplot et les graphe ci-dessus, nous pouvons classer les variables : 
# 
# > **Bien : V1, V2, V3, V4, V6, V7, V9, V10, V11, V12, V14, V17, V18, V19, V21, V27**
# 
# > **Moins Bien : V5, V8, V13, V15, V16, V20, V22, V23, V24, V25, V26, V28**
# 
# Nous supprimons aussi la colonne "Time" et gardons les colonnees "Hour" et "Hour_of_day" en considérant le niveau de dispersion. 
# 
# Cela nous permet de regarder les performances en éliminant certains variables.

# In[ ]:


# relation entre Amount et Class

def distplot(data):
    group_labels = ['fraud', 'non fraud']
    hist_data = [df[data][df["Class"]==1], df[data][df["Class"]==0]]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.5, show_rug=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', showline=True, linecolor='LightGrey', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', showline=True, linecolor='LightGrey', mirror=True)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        height=350, 
        title="histogramme de " + data,
        title_font_color="Grey",
    )
    fig.show()
    return

variables = df.iloc[:,1:29]
for variable in variables:
    distplot(variable) 


# # 3.Entraînement du modèle

# ## 3.1 Traitement le déséquilibre de données : tous les features
# > On separe les données en modèle de train et modèle de test en utilisant le paramètre **stratify** afin de rassurer que  la proportion de valeurs dans l'échantillon produit sera la même que la proportion de valeurs fournie

# In[145]:


def result_modele(classificator, XTtestSet, XTrainSet, yTestSet, yTrainSet):
    y_pred_test = classificator.predict(XTtestSet)
    target_names = ['non fraud', 'fraud']
    print('----------- test -----------')
    print(classification_report(yTestSet, y_pred_test, target_names=target_names))
    print('----------- confusion matrix -----------')
    print(confusion_matrix(yTestSet, y_pred_test))

    # train
    y_pred_train = classificator.predict(XTrainSet)
    print('----------- train ----------')
    print(classification_report(yTrainSet, y_pred_train, target_names=target_names))
    print('----------- confusion matrix -----------')
    print(confusion_matrix(yTrainSet, y_pred_train))


# In[154]:


def df_train_test_split(dataframeOrigin, testSize, randomState, dropList):
    X = dataframeOrigin.drop(columns=dropList)
    y = dataframeOrigin['Class']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=randomState, stratify=y)
    return xTrain, xTest, yTrain, yTest, y


# In[155]:


def standardScale(xTrain, xTest):
    sc = StandardScaler()
    X_train = sc.fit_transform(xTrain.values)
    X_test = sc.transform(xTest.values)
    return X_train, X_test


# In[156]:


X_train, X_test, y_train, y_test, y = df_train_test_split(df, 0.2, 0, ['Hour', 'Time', 'Class'], )


# In[157]:


X_train, X_test = standardScale(X_train, X_test)


# In[160]:


def checkProportion(y, yTrain, yTest, XTrain):
    n_echantillon = y.shape[0]
    n_echantillon_pos = y[y==0].shape[0]
    n_echantillon_neg_train = y_train.sum()/y_train.count()
    n_echantillon_neg_test = y_test.sum()/y_test.count() 
    print('nombre d\'échantillons: {}; les échantillons positifs: {:.2%}'.format(n_echantillon, 
                                                                                 n_echantillon_pos/n_echantillon))
    print('les échantillons négatifs train: {:.2%};'.format(n_echantillon_neg_train))
    print('les échantillons négatifs test: {:.2%};'.format(n_echantillon_neg_test))      
    print('nombre de variables : ', XTrain.shape[1])


# In[159]:


checkProportion(y, y_train, y_test, X_train)


# ## 3.2 Random Forest Over / Under Sampling avec tous les features
# 
# Pour traiter le déséquilibre de l'échantillon directement, on utilse la méthode de  **over et under sampling** avec tous les features
# 
# > **Over Sampling avec sampling_strategy de 0.1**
# 
# > **Under Sampling avec sampling_strategy de 0.5**

# In[23]:


def overUnderSampling(over_rate, under_rate, typeSampling):
    # definir over et under sampling 
    over = SMOTE(sampling_strategy=over_rate)
    under = RandomUnderSampler(sampling_strategy=under_rate)
    # definir pipeline et steps
    if typeSampling == 'mix':
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        return pipeline
    else : # typeSampling = 'over'
        steps_over = [('o', over)]
        pipeline_over = Pipeline(steps=steps_over)
        return pipeline_over        


# In[27]:


pipeline = overUnderSampling(0.1, 0.5, 'mix')


# In[28]:


X_res, y_res = pipeline.fit_resample(X_train, y_train)

# summarize the new class distribution
counter_y_train = Counter(y_train)
print(counter_y_train)
counter_y_res = Counter(y_res)
print(counter_y_res)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_res, y_res)


# #### 3.2.1 Recall, Precision et F1-score
# On regarde **recall**, **precision** et **f1-score** de train, test
# > **f1-score** présent une moyenen harmonie entre recall et precision qu'on a obtenu. On sait qu'il y a un compromis entre recall et precison, quan on fait plusieurs essaies, il est possilbe d'avoir 1 precision qui serait plus haut mais 1 recall qui serait plus bas. C'est difficile de comparer les modèles. Donc on calcule f1-socre, et si on a un meilleur f1-socre, le modèle est le meilleur.
# 
# > Avec les résultat de classification de test et de train, on voit bien qu'il y a **over-fitting avec f1-socre de 1, recall de 1 et precision de 1.** C'est à dire qu'on a un super modèle avec aucune erreur. 
# 
# 

# In[123]:


result_modele(clf, X_test, X_res, y_test, y_res)


# #### 3.2.2 La curbe ROC et la curbe de recall, precision sur le test
# On regard la curbe ROC sur le dataset de test
# > La curbe ROC n'est pas intéprétable car il classife très bien les négatifs. Mais les positifs ne sont pas prend en compte. Cela ne nous aide pas.

# In[111]:


def roc_curbe(clf, xTest, yTest, rocTitle):
    scores = clf.predict_proba(xTest)
    fpr, tpr, thresholds = roc_curve(yTest, scores[:,1])
    auc_ = auc(fpr, tpr)
    
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--"),
    plt.plot(fpr, tpr, color = 'red', label='ROC curve AUC=%0.5f' %(auc_)),

    plt.xlim([-0.01, 1.05]),
    plt.ylim([-0.01, 1.05]),
    plt.xlabel("False Positive Rate"),
    plt.ylabel("True Positive Rate"),
    plt.title("Receiver operating characteristic " + rocTitle),
    plt.legend(loc="lower right")
    plt.figure(figsize=(6,8))
    return scores
    


# In[112]:


scores = roc_curbe(clf, X_test, y_test, '(All feature)')


# > Il faut qu'on regarde la curbe de précision et recall avec les seuils

# In[114]:


def precision_recall_curbe(yTest, scores, figTitle):
    
    precision, recall, thresholds = precision_recall_curve(y_test, scores[:,1])
    
    columns = ['threshold', 'precision', 'recall']
    inputs = pd.DataFrame(columns=columns, dtype=np.number)

    for i in range(0, len(precision)-1):
      inputs.loc[i, 'threshold'] = thresholds[i]
      inputs.loc[i, 'precision'] = precision[i]
      inputs.loc[i, 'recall'] = recall[i]


    fig = px.line(inputs, x='recall', y='precision', hover_data=['threshold'], title='Precision / Recall Curve '+figTitle) 
    hovertemplate = 'Recall=%{x}<br>Precision=%{y}<br>Threshold=%{customdata[0]:.4f}<extra></extra>'
    fig.update_layout(autosize=False, width=800, height=600, title_x=0.5, 
                      yaxis_range=[-0.01, 1.05], xaxis_range=[-0.01, 1.05],
                      xaxis=dict(hoverformat='.4f'), yaxis=dict(hoverformat='.4f'))
    fig.update_traces(hovertemplate=hovertemplate)
    fig.show()
    return fig


# In[115]:


fig = precision_recall_curbe(y_test, scores, 'All_Features')
fig.write_html('precision_recall_curbe_all_features.html')


# ## 3.3 Random Forest Over Sampling avec tous les features
# 
# Pour traiter le déséquilibre de l'échantillon directement, on utilse la méthode de  **over et under sampling** avec tous les features
# 

# ### 3.3.1 **Avec Over Sampling de 0.5**

# In[30]:


over2 = SMOTE(sampling_strategy=0.5)
steps2 = [('o', over2)]
pipeline2 = Pipeline(steps=steps2)


# In[31]:


X_res_over, y_res_over = pipeline2.fit_resample(X_train, y_train)


# > Proportion de positifs dans le dataset 
# 

# In[32]:


# proportion de positifs dans le dataset 
y_res_over.sum()/y_res_over.count()*100


# In[33]:


clf_over = RandomForestClassifier()
clf_over.fit(X_res_over, y_res_over)


# In[102]:


result_modele(clf_over, X_test, X_res_over, y_test, y_res_over)


# ## 3.4 Entrainer un modèle avecles features choisies par AED
# 
# 
# On reprend ce qu'on a mentionné au chapitre 2.2. On va essayer d'entraîner un modèle avec des variables sélectionnées selon l'AED.
# 
# > Bien : V1, V2, V3, V4, V6, V7, V9, V10, V11, V12, V14, V17, V18, V19, V21, V27
# 
# > Moins Bien : V5, V8, V13, V15, V16, V20, V22, V23, V24, V25, V26, V28
# 
# **On élimine le variable V16 qui a une colinéarité avec V17 et les autres variables moins bien pour regarder la performance de f1-score**

# In[34]:


features_eda = standardScale(df, ['Class'])


# In[35]:


dropList = ['V5', 'V8', 'V13', 'V15', 'V16', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28', 'Hour', 'Time']
X_train_eda, X_test_eda, y_train_eda, y_test_eda, y_eda = df_train_test_split(features_eda, df, 0.2, 0, dropList)

checkProportion(y_eda, y_train_eda, y_test_eda, X_train_eda)


# ### 3.4.1 **Avec Over / under Sampling**

# In[36]:


X_res_eda, y_res_eda = pipeline.fit_resample(X_train_eda, y_train_eda)

# summarize the new class distribution
counter_y_train_eda = Counter(y_train_eda)
print(counter_y_train_eda)
counter_y_res_eda = Counter(y_res_eda)
print(counter_y_res_eda)


# In[37]:


# Random Forest sur le modèle EDA
clf_eda = RandomForestClassifier()
clf_eda.fit(X_res_eda, y_res_eda)


# Avec le résultat ce-dessus, on voit que cela n'a pas amélioré f1-socre et on a toujours le problème de l'overfitting sur notre modèle. 

# In[139]:


result_modele(clf_eda, X_test_eda, X_res_eda, y_test_eda, y_res_eda)


# In[184]:


scores_eda = roc_curbe(clf_eda, X_test_eda, y_test_eda, 'AED features')


# Avec le graphe ci-dessus, on voit que si on veut avoir un meuiller recall pour capturer plus de fraud, la précision se diminue. C'est à dire qu'on a plus de chance de se tromper.
# 
# **Intuitivement, on peut avoir un recall de 62 % qui a une précision de 96%, et le seuil est 0.96.**
# 
# Le recall et la précision dépendent ce qu'on veut obtenir : 
# 
# > Capturer plus de fraud : le recall serait plus grand. la précision et le seuil seraient plus petits
# 

# In[172]:


precision_recall_curbe(y_test_eda, scores_eda, 'EDA')


# ### 3.4.2 **Aeve Over Sampling 0.5**

# In[173]:


X_res_eda_over2, y_res_eda_over2 = pipeline2.fit_resample(X_train_eda, y_train_eda)

# summarize the new class distribution
counter_y_train_eda = Counter(y_train_eda)
print(counter_y_train_eda)
counter_y_res_eda_over2 = Counter(y_res_eda_over2)
print(counter_y_res_eda_over2)


# In[175]:


clf_eda_over2 = RandomForestClassifier()
clf_eda_over2.fit(X_res_eda_over2, y_res_eda_over2)


# In[177]:


result_modele(clf_eda_over2, X_test_eda, X_res_eda_over2, y_test_eda, y_res_eda_over2)


# ## 3.5 Entrainer un modèle avec **RFE recursive feature elimination**  
# > on choisit 18 features afin de comparer avec le résultat de AED

# In[178]:


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE

def modele_ref(xTrain, yTrain):
    svc_18 = SVC(kernel="linear")
    selector_18 = RFE(estimator=svc_18, n_features_to_select=18, step=1)
    selector_18 = selector_18.fit(xTrain, yTrain)
    # resultats
    support = selector_18.support_
    ranking = selector_18.ranking_
    featureName = selector_18.feature_names_in_
    
    # zip ranking et nom des features
    selector_zip = zip(featureName, ranking)
    list_rank_first = []
    list_vars_keep  = []
    
    for var in selector_zip:
      if var[1] == 1:
        list_rank_first.append(var)
        list_vars_keep.append(var[0])

    list_vars_keep.append('Class')

    return list_vars_keep, selector_18


# In[180]:


list_var_keep, selector_18 = modele_ref(X_res, y_res)


# In[ ]:


selector_18


# In[ ]:


selector_18.ranking_, selector_18.feature_names_in_


# In[ ]:


df_scaled = standardScale(df, ['Class'])
list_vars = list(df)
list_drop = list(set(list_vars) - set(list_vars_keep))

X_train_ref_18, X_test_ref_18, y_train_ref_18, y_test_ref_18, y_ref_18 = df_train_test_split(df_scaled, df, 0.2, 0, list_drop)


# ### 3.5.1 Avec Over / under Sampling
# 
# 
# 

# In[ ]:


X_res_ref_18, y_res_ref_18 = pipeline.fit_resample(X_train_ref_18, y_train_ref_18)

# Random Forest sur le modèle REF
clf_ref_18 = RandomForestClassifier()
clf_ref_18.fit(X_res_ref_18, y_res_ref_18)


# In[ ]:


result_modele(clf_ref_18, X_test_ref_18, X_res_ref_18, y_test_ref_18, y_res_ref_18):


# In[ ]:


scores_ref_18 = roc_curbe(clf_ref_18, X_test_ref_18, y_test_ref_18, '(RFE feature)')


# In[ ]:


precision_recall_curbe(y_test_ref_18, scores_ref_18, 'RFE features')


# ### 3.5.2 Avec Over Sampling de 0.5
# 

# In[ ]:


X_res_ref_18_over, y_res_ref_18_over = pipeline2.fit_resample(X_train_ref_18, y_train_ref_18)

# Random Forest sur le modèle REF
clf_ref_18_over = RandomForestClassifier()
clf_ref_18_over.fit(X_res_ref_18_over, y_res_ref_18_over)


# In[ ]:


result_modele(clf_ref_18_over, X_test_ref_18, X_res_ref_18_over, y_test_ref_18, y_res_ref_18_over)


# # 4.Optimisation Modèle : Tuning avec RandomizedSearchCV
# 
# > Trouver les meilleurs hyperparamètres afin de traiter le problème d'overfitting
# 
# - n_estimators：nombre d'arbre utilisé
# - max_features：nombre de feature utilisé, par exemple : “auto”, “sqrt”, “log2”
#     - “auto”>> max_features=sqrt(n_features).
#     - “sqrt”>> then max_features=sqrt(n_features) (same as “auto”).
#     - “log2”>> then max_features=log2(n_features).
# 
# - max_depth(default=None): la profondeur maximun (important)
# - min_samples_split(default=2)
# - min_samples_leaf(default=1)

# In[38]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [50,100,200,400,800]
max_features = ['auto']
max_depth = [3,5,7,10,12]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
random_grid


# ## 4.1 Avec tous les features 
# 

# In[46]:


def randomizedSearchCV(xTrain, yTrain, Niter, cvFold, randomGrid):
    forest = RandomForestClassifier(random_state=42)
    
    rf_random = RandomizedSearchCV(estimator = forest, param_distributions=randomGrid,
                              n_iter=Niter, cv=cvFold, verbose=2, random_state=42, n_jobs=-1)

    rf_random.fit(xTrain, yTrain)
    best_params = rf_random.best_params_
    print(best_params)
    return rf_random


# In[47]:


def randomForestFitBestPara(randomBest, xTrain, yTrain):
    forest = RandomForestClassifier(max_depth=randomBest.best_params_['max_depth'], 
                                 max_features=randomBest.best_params_['max_features'], 
                                 min_samples_leaf=randomBest.best_params_['min_samples_leaf'], 
                                 min_samples_split=randomBest.best_params_['min_samples_split'],
                                 n_estimators=randomBest.best_params_['n_estimators'])
    
    forest_fit = forest.fit(xTrain, yTrain)
    
    return forest_fit


# ### Over / Under Sampling Mix

# In[48]:


randBest = randomizedSearchCV(X_res, y_res, 10, 3, random_grid)


# In[51]:


forest_fit = randomForestFitBestPara(randBest, X_res, y_res)


# In[52]:


result_modele(forest_fit, X_test, X_res, y_test, y_res)


# ### Over Sampling de 0,5 avec n_iter de 50 et cross validation de 4

# In[53]:


randBest = randomizedSearchCV(X_res_over, y_res_over, 50, 4, random_grid)


# In[54]:


forest_fit = randomForestFitBestPara(randBest, X_res_over, y_res_over)


# In[55]:


result_modele(forest_fit, X_test, X_res_over, y_test, y_res_over)


# ### Over Sampling de 0,5 avec n_iter de 100 et cross validation de 4

# In[120]:


randBest_over_100 = randomizedSearchCV(X_res_over, y_res_over, 100, 4, random_grid)


# In[121]:


forest_fit_over_100 = randomForestFitBestPara(randBest_over_100, X_res_over, y_res_over)


# In[122]:


result_modele(forest_fit_over_100, X_test, X_res_over, y_test, y_res_over)


# ### Sampling Mix avec n_iter de 100 et cross validation de 4

# In[117]:


randBest = randomizedSearchCV(X_res, y_res, 100, 4, random_grid)


# In[118]:


forest_fit_100 = randomForestFitBestPara(randBest, X_res, y_res)


# In[119]:


result_modele(forest_fit_100, X_test, X_res, y_test, y_res)


# ### Over Sampling de 0,5 avec n_iter de 10 et cross validation de 3

# In[43]:


randBest = randomizedSearchCV(X_res_over, y_res_over, 10, 3, random_grid)


# In[45]:


forest_over = RandomForestClassifier(max_depth=rf_random_over.best_params_['max_depth'], 
                                 max_features=rf_random_over.best_params_['max_features'], 
                                 min_samples_leaf=rf_random_over.best_params_['min_samples_leaf'], 
                                 min_samples_split=rf_random_over.best_params_['min_samples_split'],
                                 n_estimators=rf_random_over.best_params_['n_estimators'],
                                 )
forest_over = forest_over.fit(X_res_over, y_res_over)


# In[185]:


result_modele(forest_over, X_test, X_res_over, y_test, y_res_over)


# ## 4.2 Avec les features AED
# 

# ### Over / Under Sampling

# In[48]:


# X_res_eda, y_res_eda = pipeline.fit_resample(X_train_eda, y_train_eda)
rf_random_eda = rf_random.fit(X_res_eda, y_res_eda)
rf_random_eda.best_params_


# In[49]:


forest_eda = RandomForestClassifier(max_depth=rf_random_eda.best_params_['max_depth'], 
                                 max_features=rf_random_eda.best_params_['max_features'], 
                                 min_samples_leaf=rf_random_eda.best_params_['min_samples_leaf'], 
                                 min_samples_split=rf_random_eda.best_params_['min_samples_split'],
                                 n_estimators=rf_random_eda.best_params_['n_estimators'])

forest_eda_fit = forest_eda.fit(X_res_eda, y_res_eda)


# In[57]:


result_modele(forest_eda_fit, X_test_eda, X_res_eda, y_test_eda, y_res_eda)


# ### Over Sampling

# In[60]:


rf_random_eda_over = rf_random.fit(X_res_eda_over2, y_res_eda_over2)
rf_random_eda_over.best_params_


# In[52]:


forest_eda_over = RandomForestClassifier(max_depth=rf_random_eda_over.best_params_['max_depth'], 
                                 max_features=rf_random_eda_over.best_params_['max_features'], 
                                 min_samples_leaf=rf_random_eda_over.best_params_['min_samples_leaf'], 
                                 min_samples_split=rf_random_eda_over.best_params_['min_samples_split'],
                                 n_estimators=rf_random_eda_over.best_params_['n_estimators'])

forest_eda_over_fit = forest_eda_over.fit(X_res_eda_over2, y_res_eda_over2)


# In[55]:


result_modele(forest_eda_over_fit, X_test_eda, X_res_eda_over2, y_test_eda, y_res_eda_over2)


# # 5.Optimisation Modèle : Tuning avec hyperOpt bayesienne
# > Au lieu de chercher au hasard, j'ai essayé la méthode Bayesienne pour chercher les ranges des hyper paramètres

# In[72]:


# Définir l'espace qui contient les hyper-paramètres que j'ai cherché à optimiser
space  = [Integer(1, 12, name='max_depth'),
          Integer(50, 800, name='n_estimators'),
          Integer(2, 50, name='min_samples_split'),
          Integer(1, 50, name='min_samples_leaf')]

# Recevoir les keyword paramètres
@use_named_args(space)
def objective(**params):
    clf.set_params(**params)
    return -np.mean(cross_val_score(clf, X_res, y_res, cv=5, n_jobs=-1,scoring="f1"))

# Pour dataset over sampling
@use_named_args(space)
def objective2(**params):
    clf_over.set_params(**params)
    return -np.mean(cross_val_score(clf_over, X_res_over, y_res_over, cv=5, n_jobs=-1,scoring="f1"))


# In[128]:


# 20 call
from skopt import gp_minimize
res_gp_20 = gp_minimize(objective, space, n_calls=20, random_state=0)
"Best score=%.4f" % res_gp_20.fun


# In[127]:


print("""Best parameters:
- max_depth=%d
- n_estimators=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_20.x[0], res_gp_20.x[1], 
                            res_gp_20.x[2], res_gp_20.x[3])
)


# In[59]:


from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
"Best score=%.4f" % res_gp.fun


# In[60]:


print("""Best parameters:
- max_depth=%d
- n_estimators=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 
                            res_gp.x[2], res_gp.x[3])
     )


# In[78]:


res_gp_over = gp_minimize(objective2, space, n_calls=10, random_state=0)
"Best score=%.4f" % res_gp_over.fun


# In[79]:


print("""Best parameters:
- max_depth=%d
- n_estimators=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_over.x[0], res_gp_over.x[1], 
                            res_gp_over.x[2], res_gp_over.x[3])
     )


# ## 5.1 Training avec le modèle de tous les features

# In[129]:


forest_hyperOpt_20 = RandomForestClassifier(max_depth=10, 
                                 max_features='auto', 
                                 min_samples_leaf=1, 
                                 min_samples_split=2,
                                 n_estimators=50)

forest_hyperOpt_fit_20 = forest_hyperOpt_20.fit(X_res, y_res)


# In[130]:


result_modele(forest_hyperOpt_20, X_test, X_res, y_test, y_res)


# In[131]:


forest_hyperOpt_fit_20_over = forest_hyperOpt_20.fit(X_res_over, y_res_over)


# In[132]:


result_modele(forest_hyperOpt_fit_20_over, X_test, X_res_over, y_test, y_res_over)


# In[125]:


forest_hyperOpt_20call = RandomForestClassifier(max_depth=res_gp_20.x[0], 
                                 max_features='auto', 
                                 min_samples_leaf=res_gp_20.x[3], 
                                 min_samples_split=res_gp_20.x[2],
                                 n_estimators=res_gp_20.x[1])

forest_hyperOpt_fit_20call = forest_hyperOpt_20call.fit(X_res, y_res)


# In[126]:


result_modele(forest_hyperOpt_fit_20call, X_test, X_res, y_test, y_res)


# In[61]:


forest_hyperOpt = RandomForestClassifier(max_depth=res_gp.x[0], 
                                 max_features='auto', 
                                 min_samples_leaf=res_gp.x[3], 
                                 min_samples_split=res_gp.x[2],
                                 n_estimators=res_gp.x[1])

forest_hyperOpt_fit = forest_hyperOpt.fit(X_res, y_res)


# In[62]:


result_modele(forest_hyperOpt_fit, X_test, X_res, y_test, y_res)


# ## 5.2 Training avec le modèle AED qui contient 18 features

# In[81]:


forest_over_hyperOpt_ = RandomForestClassifier(max_depth=res_gp_over.x[0], 
                                 max_features='auto', 
                                 min_samples_leaf=res_gp_over.x[3], 
                                 min_samples_split=res_gp_over.x[2],
                                 n_estimators=res_gp_over.x[1])

forest_over_hyperOpt_fit = forest_over_hyperOpt_.fit(X_res_over, y_res_over)


# In[82]:


result_modele(forest_hyperOpt_fit, X_test, X_res_over, y_test, y_res_over)


# In[ ]:




