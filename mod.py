import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
from _datetime import datetime
import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import pymongo


ordinal_var = {"C1": {"Fluvial": 0, "Other": 1, "Torrential": 2},
              "C5":{"Plain": 0, "Other": 1, "Mountain": 2},
              "C6":{"Almost straight": 0, "Sinuous": 1, "Very sinuous": 1,"Extremely sinuous":2},
              "C7":{"Rock": 0, "Cohesive soil": 1, "Cohesionless soil": 2},
              "B8":{"Triangular-nosed" :0, "Circular or oblong":1, "Rectangular":2},
              "B10":{"No" :0, "Yes":1},
              "H11":{"No" :0, "Yes":1},
              "H12":{"No" :0, "Yes":1},
              "I13":{"No" :0, "Yes":1},
              "I14":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3,'Very Bad':4},
              "I15":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3,'Very Bad':4},
              "I16":{"No" :0, "Yes":1},
              "I17":{"No" :0, "Yes":1},
              "I18":{'Very good':0, 'Good':1, 'Fair':2, 'Poor':3}}

#TODO color
def gauge(value):
    """
    utilise la librairie st-echart
    pour inserer des graphique echart(js)
    link echart: https://echarts.apache.org/examples/en/index.html
    """
    options = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Prediction probality",
                "type": "gauge"
                ,
                "axisLine": {
                    "lineStyle": {
                        "color": [
                            [0.30, '#37a2da'],
                            [0.70, '#b2c609'],
                            [1, "#ff5733"]
                        ]
                    },
                },
                "progress": {"show": "true", "width": 0},
                "detail": {"valueAnimation": "true", "formatter": "{value}%"},
                "data": [{"value": value, "name": "Risk"}],
                "pointer": {
                    "itemStyle": {
                        "distance": 100,
                        "color": 'auto',
                        "fontSize": 80
                    }
                }, "axisLabel": {
                    "distance": 7
                    },
            },
        ],
    }

    st_echarts(options=options, width="100%", key=0)

def user_input_features():
    i20 = st.sidebar.selectbox("Bridge element ", ["Mur/Culée (Wall/Abutment)", "Pile(Pier))"]),
    I19 = st.sidebar.selectbox("Country code", ['France', 'Other'])
    if I19 == "France":
        i20 = st.sidebar.text_input('Pk')
    if I19 == "Other":
        i20 = st.sidebar.text_input('Country code + N')
    C1 = st.sidebar.selectbox('Flow type (C1)',('Fluvial','Other','Torrential'))
    C2 = st.sidebar.number_input('Slope of riverbed (%) (C2)', min_value=0.01, max_value=3.08)
    C3 = st.sidebar.number_input('Flood flow (C3)', min_value=2.09, max_value=5457.26)
    C4 = st.sidebar.number_input('WV/WC (C4)', min_value=1.52, max_value=226.68)
    C5 = st.sidebar.selectbox('Topography (C5)',('Plain','Other','Mountain'))
    C6 = st.sidebar.selectbox('Sinuosity (C6)',('Almost straight', 'Sinuous', 'Extremely sinuous'))
    C7 = st.sidebar.selectbox('Riverbed material (C7)',('Rock','Cohesive soil', 'Cohesionless soil'))
    B8 = st.sidebar.selectbox('Pier shape (B8)',('Triangular-nosed', 'Circular or oblong','Rectangular'))
    B9 = st.sidebar.selectbox('Foundation type (B9)',('Concrete/ciment','Timber piles','Caisson'))
    B10 = st.sidebar.selectbox('Existence of foundation scour countermeasures (B10)',('No','Yes'))
    H11 = st.sidebar.selectbox('Scour history (H11)',('No','Yes'))
    H12 = st.sidebar.selectbox('Flood history (H12)',('No','Yes'))
    I13 = st.sidebar.selectbox('Susceptible of scour (I13)',('No','Yes'))
    I14 = st.sidebar.selectbox('Channel rating (I14)',('Very good', 'Good', 'Fair', 'Poor','Very Bad'))
    I15 = st.sidebar.selectbox('Riverbank rating (I15)',('Very good', 'Good', 'Fair', 'Poor','Very Bad'))
    I16 = st.sidebar.selectbox('Existence of dislocation or deformation (I16)',('No','Yes'))
    I17 = st.sidebar.selectbox('Existence of local scour (I17)',('No','Yes'))
    I18 = st.sidebar.selectbox('Rating of other damages (corrosion, timber piles degradation, cracks, etc.) (I18)', ('Very good', 'Good', 'Fair', 'Poor'))


    data = {'C1': C1,
            'C2': C2,
            'C3': C3,
            'C4': C4,
            'C5': C5,
            'C6': C6,
            'C7': C7,
            'B8': B8,
            'B9': B9,
            'B10': B10,
            'H11': H11,
            'H12': H12,
            'I13': I13,
            'I14': I14,
            'I15': I15,
            'I16': I16,
            'I17': I17,
            'I18': I18 }
    features = pd.DataFrame(data, index=[0])
    return features,i20

def pipeline(input_df):
    """
    preprocessing des valeurs de l'utilisateur
    pour les mettre au format du model (One-hot encoding)
    return tupple
    (dforiginal + user input, y du df original)
    """
    df_num = input_df.replace(ordinal_var)
    dforiginal = pd.read_excel('../pier.xlsx', engine='openpyxl')
    y = dforiginal["Risk_bin"]
    df_X = dforiginal.drop(columns=['Risk_bin'])
    df_num.columns = df_X.columns
    df = pd.concat([df_num,df_X],axis=0)
    dummy = pd.get_dummies(df["Fdt_type"], prefix="Fdt_type")
    df = pd.concat([df,dummy], axis=1)
    del df["Fdt_type"]
    col = ['Flow_type', 'Slope_bed(%)', 'Flow', 'WV/WC', 'Topo', 'Sinuosity', 'Riv_material', 'Scpt_scour', 'Flood_hstry', 'Scour_hstry', 'Channel_rating', 'Bank_rating', 'Pier_shape', 'Ctms_foundation', 'Masonry_YN', 'Local_scour_YN', 'Other_damages', 'Fdt_type_Caisson', 'Fdt_type_Concrete/ciment', 'Fdt_type_Timber piles']
    df.columns = col
    return df, y

def pred(df):
    """
    recupere la premiere ligne du df
    load et run le model avec celle-ci
    return tupple
    ([proba 0,proba 1], classe predite)
    """
    clf = joblib.load('../xgb_clf.pkl')
    return clf.predict(df.head(1))[0], clf.predict_proba(df.head(1))[0], clf

def insert_mongo(df_m, classe, proba,val_bol):
    """
    recupere les valeurs entre par l'user, la classe predite,les probalité des deux classes
    et insert cell-ci daqns la bdd mongo
    return None
    """
    dictionary = dict(zip(df_m["columns"], [str(x) for x in df_m["data"][0]]))
    d = {**{"datetime":datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    "classe_predite":str(classe),"proba":str(proba),
    "avis_user": val_bol
    },**dictionary}
    pymongo.MongoClient()["sncf"]["model_historique"].insert_one(d)

def form_data():
    """
    recupere les données du formulaire en provenance de la bdd
    pour les retourner sous le format dataframe
    return DATAFRAME
    """
    for x in pymongo.MongoClient()["sncf"]["formu_user"].find({}, {"_id": 0}):
        df = pd.DataFrame.from_dict([x])
    return df

def model_historique():
    """
      recupere les données de prediction en provenance de la bdd
      pour les retourner sous le format dataframe
      return DATAFRAME
      """
    val = []
    for x in pymongo.MongoClient()["sncf"]["model_historique"].find({}, {"_id": 0}):
        val.append(x)
    return pd.DataFrame(val)

def download_data():
    """
      recupere les données en provenance de la bdd
      pour les fournir a l'utilisateur
      sous forme de csv
      return None
      """
    csv = form_data().to_csv().encode('utf-8')
    mode_data = model_historique().to_csv().encode('utf-8')
    st.download_button(
   "Press to Download form data",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
    )
    st.download_button(
   "Press to Download model historique",
   mode_data,
   "file.csv",
   "text/csv",
   key='download-csv'
    )

def plot1(xgb):
    st.write('Feature importance')
    st.pyplot(xgboost.plot_importance(xgb).figure)


def plot2(xgb):
    help_input = '''This is the line1\n
    This is the line 2\n
    This is the line 3'''

    st.info('TREEEXPLAINER\n\n Ce graphique represente les colonnes qui ont le plus influencer le model\n\n'
            'À gauche vous trouverez la caractéristique ainsi que sa valeur\n\n'
            "À droite l'impact de celle-ci sur le model \n\n - **BlEU **: reduit le risque \n\n - **ROUGE **: augmente le risque")
    explainer = shap.TreeExplainer(xgb, st.session_state["row"])
    shap_values = explainer(st.session_state["row"].head(1))
    fig = plt.figure(figsize=(3.3, 2.3))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig,help=help_input)

def plot3(xgb):
    st.write('Decision tree')
    fig, ax = plt.subplots(figsize=(30, 30))
    xgboost.plot_tree(xgb, num_trees=4, ax=ax)
    st.pyplot(fig)

def plot4(xgb):
    st.write('Confusion Matrix')
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
    import numpy as np
    fig, ax = plt.subplots(figsize=(30, 30))
    cf_matrix = confusion_matrix(st.session_state['y'], xgb.predict(st.session_state["row"][:-1]))
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues', annot_kws={'size': 90})
    st.pyplot(fig)

