import streamlit
import ast
import pickle
import sre_constants
import joblib
import pymongo
from mod import *
import pandas as pd
from PIL import Image
import time
import pymongo
import datetime

db = MongoClient("mongodb+srv://sncf:sncf2022@cluster0.rlxfwkw.mongodb.net/?retryWrites=true&w=majority")
image = Image.open('images/sncf_logo.png')
st.image(image)
st.title('XGBoost classifier for bridge scour risk prediction')
st.text \
    ('This classfieir is trained and tested by using an SNCF \ndataset with the aim to predict the bridge pier scour risk.')
di = {
    "I20_eng":{
        "question":"Bridge element",
        "reponse": ["Wall/Abutment", "Pier"]},
    "I20_fr":{
        "question": "Elément du pont",
        "reponse":["Mur/Culée","Pile"]
    },

    "I19_eng": {
        "question": "Country code",
        "reponse": ['Fr', 'Other']},
    "I19_fr": {
        "question": "Code pays",
        "reponse": ["Fr", "Autre"]
    },

    "C1_eng": {
        "question": 'Flow type (C1)',
        "reponse": ('Fluvial', 'Other', 'Torrential')},
    "C1_fr": {
        "question": "Type de flux",
        "reponse": ["Fluvial", "Autre","Torrentiel"]
    },

    "C2_eng": {
        "question": "Slope of riverbed (%) (C2)",
        "reponse": (0.01,3.08)},
    "C2_fr": {
        "question": "Pente du lit de la rivière (%) (C2)",
        "reponse": (0.01,3.08)
    },

    "C3_eng": {
        "question": 'Flood flow (C3)',
        "reponse": (2.09, 5457.26)},
    "C3_fr": {
        "question": "Débit de crue (C3)",
        "reponse": (2.09, 5457.26)
    },

    "C4_eng": {
        "question": 'WV/WC (C4)',
        "reponse": (2.09, 5457.26)},
    "C4_fr": {
        "question": "WV/WC (C4)",
        "reponse": (2.09, 5457.26)
    },

    "C5_eng": {
        "question": 'Topography (C5)',
        "reponse": ('Plain','Other','Mountain')},
    "C5_fr": {
        "question": "Topographie",
        "reponse": ["Plaine", "Autre", 'Montagne']
    },

    "C6_eng": {
        "question": 'Sinuosity (C6)',
        "reponse": ('Almost straight', 'Sinuous', 'Extremely sinuous')},
    "C6_fr": {
        "question": "Sinuosité (C6)",
        "reponse": ["Presque droit", "Sinueux", 'Extrêmement sinueux']
    },

    "C7_eng": {
        "question": 'Riverbed material (C7)',
        "reponse": ('Rock', 'Cohesive soil', 'Cohesionless soil')},
    "C7_fr": {
        "question": "Matériau du lit de la rivière (C7)",
        "reponse": ["Rock", "Soil cohésif", 'Soil sans cohésion']
    },

    "B8_eng": {
        "question": 'Pier shape (B8)',
        "reponse": ('Triangular-nosed', 'Circular or oblong', 'Rectangular')},
    "B8_fr": {
        "question": "Forme de jetée",
        "reponse": ["Triangulaire", "Circulaire ou oblong", 'Rectangulaire']
    },

    "B9_eng": {
        "question": 'Foundation type (B9)',
        "reponse": ('Concrete/ciment','Timber piles','Caisson')},
    "B9_fr": {
        "question": "Type de fondation",
        "reponse": ["Béton/ciment", "Pieux bois", 'Caisson']
    },

    "B10_eng": {
        "question": 'Existence of foundation scour countermeasures (B10)',
        "reponse": ('No', 'Yes')},
    "B10_fr": {
        "question": "Existence de contre-mesures contre l'affouillement des fondations",
        "reponse": ["Non", "Oui"]
    },

    "H11_eng": {
        "question": 'Scour history (H11)',
        "reponse": ('No', 'Yes')},
    "H11_fr": {
        "question": "Historique d'affouillement",
        "reponse": ["Non", "Oui"]
    },

    "H12_eng": {
        "question": 'Flood history (H12)',
        "reponse": ('No', 'Yes')},
    "H12_fr": {
        "question": "Historique des crues",
        "reponse": ["Non", "Oui"]
    },

    "I13_eng": {
        "question": 'Susceptible of scour (I13)',
        "reponse": ('No', 'Yes')},
    "I13_fr": {
        "question": "Sensible à l'affouillement (I13)",
        "reponse": ["Non", "Oui"]
    },

    "I14_eng": {
        "question": 'Channel rating (I14)',
        "reponse": ('Very good', 'Good', 'Fair', 'Poor', 'Very Bad')},
    "I14_fr": {
        "question": "Evaluation de la chaîne (I14)",
        "reponse": ["Très bien", "Bien", "Passable", "Médiocre", "Très mauvais"]
    },

    "I15_eng": {
        "question": 'Riverbank rating (I15)',
        "reponse": ('Very good', 'Good', 'Fair', 'Poor', 'Very Bad')},
    "I15_fr": {
        "question": "Evaluation des berges",
        "reponse": ["Très bien", "Bien","Passable", "Médiocre","Très mauvais"]
    },

    "I16_eng": {
        "question": 'Existence of dislocation or deformation (I16)',
        "reponse": ('No', 'Yes')},
    "I16_fr": {
        "question": "Existence de luxation ou de déformation (I16)",
        "reponse": ('Oui', 'Non')
    },

    "I17_eng": {
        "question": 'Existence of local scour (I17)',
        "reponse": ('No', 'Yes')},
    "I17_fr": {
        "question": "Existence d'affouillement local",
        "reponse": ('Oui', 'Non')
    },

    "I18_eng": {
        "question": 'Rating of other damages (corrosion, timber piles degradation, cracks, etc.) (I18)',
        "reponse": ('Very good', 'Good', 'Fair', 'Poor','Very Bad')},
    "I18_fr": {
        "question": "Évaluation des autres dommages corrosion, dégradation des pieux en bois, fissures, etc.",
        "reponse":["Très bien", "Bien", "Passable", "Médiocre", "Très mauvais"]
    }
}

def user_input_features2():
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


st.session_state["lang"] = st.sidebar.radio("Language",["fr","eng"])

def widget(func,id,di=di, lang = st.session_state["lang"] ):
    if func == st.sidebar.number_input:
        return func(di[id + lang]["question"], *di[id + lang]["reponse"], key=id)
    if st.session_state["lang"] == "fr":
        temp_key = func(di[id + lang]["question"], di[id + lang]["reponse"], key=id)
        return di[id +"eng"]["reponse"][di[id + lang]["reponse"].index(temp_key)]
    return func(di[id + lang]["question"], di[id + lang]["reponse"], key=id)

def user_input_features():
    I20 = widget(st.sidebar.selectbox, "I20_")
    I19 = widget(st.sidebar.selectbox, "I19_")
    if I19 == "Fr":
        i20 = st.sidebar.text_input('Pk', key="pk")
    if I19 == "Other":
        i20 = st.sidebar.text_input('Country code + N',key="other")
    C1 = widget(st.sidebar.selectbox, "C1_")
    C2 = widget(st.sidebar.number_input, "C2_")
    C3 = widget(st.sidebar.number_input, "C3_")
    C4 = widget(st.sidebar.selectbox, "C4_")
    C5 = widget(st.sidebar.selectbox, "C5_")
    C6 = widget(st.sidebar.selectbox, "C6_")
    C7 = widget(st.sidebar.selectbox, "C7_")
    B8 = widget(st.sidebar.selectbox, "B8_")
    B9 = widget(st.sidebar.selectbox, "B9_")
    B10 = widget(st.sidebar.selectbox, "B10_")
    H11 = widget(st.sidebar.selectbox, "H11_")
    H12 = widget(st.sidebar.selectbox, "H12_")
    I13 = widget(st.sidebar.selectbox, "I13_")
    I14 = widget(st.sidebar.selectbox, "I14_")
    I15 = widget(st.sidebar.selectbox, "I15_")
    I16 = widget(st.sidebar.selectbox, "I16_")
    I17 = widget(st.sidebar.selectbox, "I17_")
    I18 = widget(st.sidebar.selectbox, "I18_")
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

st.session_state["user_data"],st.session_state["id"] = user_input_features()
st.write(st.session_state["user_data"])

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    text-align:center;
}
</style>""", unsafe_allow_html=True)


if any(st.session_state["user_data"]):
    st.session_state["row"], st.session_state['y'] = pipeline(st.session_state["user_data"])
    st.write('  ')
    st.write('  ')
    st.write('  ')
    st.write('  ')
    button1 = st.button('Run model')

    if st.session_state.get('button') != True:
        st.session_state['button'] = button1

    if st.session_state['button'] == True:
        st.info('souhaitez vous archiver les données dans la database ?')
        oui = st.checkbox('OUI', value=False,key= "save")
        non = st.checkbox("NON", value=False,key= "notsave")

        if oui or non != False :
            classe, proba, xgb = pred(st.session_state["row"].head(1))

        if oui or non != False:
            time.sleep(0.5)
            gauge(round(proba[1] * 100, 2))
            time.sleep(0.5)
            if proba[1] < 0.25:
                st.success(f'RISQUE BAS \n\nestimation du risque a {round(proba[1]*100,2)}%')
            elif proba[1] < 0.70:
                st.info(f'RISQUE MOYEN \n\nestimation du risque a {round(proba[1]*100,2)}%')
            elif proba[1] < 1:
                st.error(f'RISQUE ELEVEE \n\nestimation du risque a {round(proba[1] * 100, 2)}%')
            st.write('*'*100)

        if "classe" in locals():
            st.header("Etes vous d'accord avec la prediction?")
            yes = st.checkbox('OUI', value=False)
            no = st.checkbox("NON", value=False)
            if yes and oui != False :
                insert_mongo(streamlit.session_state['user_data'].to_dict(orient="split"), classe, proba,"pos")
            if no and oui!= False:
                insert_mongo(streamlit.session_state['user_data'].to_dict(orient="split"), classe, proba,"neg")
            plot2(xgb)

ex = st.expander("Help us with this form to improve this model")

with ex:
    with st.form("Formulaire"):
        q_1 = st.text_input("Dans quel PRI travaillez-vous ?")
        q_2 = st.text_input('enter you name')
        q_3 = st.text_input('enter you entity')
        q_4 = st.text_input('enter you position')
        q_5 = st.radio('quel age avez vous ?', ["" ,"moins de 30 ans", "entre 30 et 40 ans", "plus de 50 ans", "je ne veux pas le dire"])
        q_6 = st.radio("Depuis combien de temps travaillez-vous dans l'inspection des ouvrages d'art ?",
              ["" ,"moins de 5 ans", "entre 5 et 10 ans", "plus de 10 ans", "je ne veux pas le dire"])

        st.subheader \
            ("D'après votre expérience, pouvez vous indiquer l'ordre d'importance des caractéristiques de l'ouvrage pour évaluer le risque de la fondation ?\n(1 : plus important ; 4 : moins important)")
        li_important = ["" ,1, 2, 3 ,4]
        q_7_1 = st.radio("Type de la fondation (pieux bois, caisson métallique, superficielle, béton, etc.) ", li_important)
        q_7_2 = st.radio("Forme de la pile (circulaire, retangulaire, etc.)", li_important)
        q_7_3 = st.radio("Largeur de la pile ", li_important)
        q_7_4 = st.radio \
            ("Présence d'une protection sur la fondation (palplanche, ceinturage en pieux bois, gabion, enrochement, etc.)"
            ,li_important)

        st.subheader \
            ("D'après votre expérience, pouvez vous indiquer l'ordre d'importance de l'environnement (hydraulique et géographique) pour évaluer le risque de la fondation?\n (1 : plus important ; 4 : moins important)")
        q_8_1 = st.radio("Type d'écoulement (fluvial, torrentiel, etc.)" ,li_important)
        q_8_2 = st.radio("Paramètres hydrauliques (profondeur de l'eau, vitesse de courant, etc.)  " ,li_important)
        q_8_3 = st.radio \
            ("Paramètres hydromorphologiques (pente du lit, rapport entre la largeur du lit majeur et du lit mineur, etc.)"
            ,li_important)
        q_8_4 = st.radio("Sinuosité du cours d'eau" ,li_important)
        q_8_5 = st.radio("Topographie" ,li_important)
        q_8_6 = st.radio("Matériaux du fond du lit (argile, roche, sables,etc.) " ,li_important)

        st.subheader \
            ("D'après votre expérience, pouvez vous indiquer l'ordre d'importance des historiques de l'ouvrage pour évaluer le risque de la fondation ?  (1 : plus important ; 4 : moins important)")
        q_9_1 = st.radio("Historique d'affouillement " ,li_important)
        q_9_2 = st.radio("Historique d'inondation " ,li_important)
        q_9_3 = st.radio("Historique d'une surveillance renforcée (concernant les FSA)" ,li_important)

        st.subheader \
            ("D'après votre expérience, pouvez vous indiquer l'ordre d'importance des constatations sur l'ouvrage pendant l'inspection pour évaluer le risque de la fondation ?  (1 : plus important ; 4 : moins important)")
        q_10_1 = st.radio("Constatations dans le cours d'eau (embâcle, fosse, vortex, végétation, etc.) " ,li_important)
        q_10_2 = st.radio("Constatations sur la berge (végétation, glissement, etc.)" ,li_important)
        q_10_3 = st.radio \
            ("Constations sur la fondation (affouillement local dans la fondation, corrosion, dégradations des pieux bois, etc. )"
            ,li_important)
        q_10_4 = st.radio("Constatations sur les murs et voûtes  (fissure, dislocation, etc.)" ,li_important)
        q_10_5 = st.radio("Suivi topographique (mouvement de la fondation)" ,li_important)
        q_10_6 = st.radio \
            ("Constations sur la protection de la fondation ( déformation de gabion, tête cassée des enceintes ou rideaux)"
            ,li_important)

        st.subheader \
            ("D'après votre expérience, pouvez vous indiquer l'ordre d'importance de ces quatre catégories générales suivants pour évaluer le risque de la fondation ? (1 : plus important ; 4 : moins important)")
        q_11_1 = st.radio("Caractéristique de l'ouvrage " ,li_important)
        q_11_2 = st.radio("Environnement (hydraulique et géographique)" ,li_important)
        q_11_3 = st.radio("Historique de l'ouvrage " ,li_important)
        q_11_4 = st.radio("Constatations sur l'ouvrage pendant l'inspection " ,li_important)

        #TODO champs text avis features
        st.subheader \
            ("Avez-vous des commentaires à ajouter ?\n")
        q_12_1 = st.text_area("", placeholder="faites nous vos propositions ici")
        submitted = st.form_submit_button("Submit")


if submitted:
    form_input = {
    "datetime":datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    "q_1":q_1,
    "q_2":q_2,
    "q_3":q_3,
    "q_4":q_4,
    "q_5":q_5,
    "q_6":q_6,
    "q_7_1":q_7_1,
    "q_7_2":q_7_2,
    "q_7_3":q_7_3,
    "q_7_4":q_7_4,
    "q_8_1":q_8_1,
    "q_8_2":q_8_2,
    "q_8_3":q_8_3,
    "q_8_4":q_8_4,
    "q_8_5":q_8_5,
    "q_8_6":q_8_6,
    "q_9_1":q_9_1,
    "q_9_2":q_9_2,
    "q_9_3":q_9_3,
    "q_10_1":q_10_1,
    "q_10_2":q_10_2,
    "q_10_3":q_10_3,
    "q_10_4":q_10_4,
    "q_10_5":q_10_5,
    "q_10_6":q_10_6,
    "q_11_1":q_11_1,
    "q_11_2":q_11_2,
    "q_11_3":q_11_3,
    "q_11_4":q_11_4}
    st.success(f"formulaire envoyé")
    db["sncf"]["formu_user"].insert_one(form_input)

st.header("References")
st.write('Wang, T., Reiffsteck, P., Chevalier, C., Zhu, Z., Chen, C.-W., & Schmidt, F. (2022). \n A novel extreme gradient boosting algorithm based model for predicting the scour risk around bridge piers: application to French railway bridges. European Journal of Environmental and Civil Engineering, 1–19.')
st.write("- [Article](https://doi.org/10.1080/19648189.2022.2072957)")
st.write('Wang, T., Reiffsteck, P., Chevalier, C., Chen, C.-W., Schmidt, F., Maintenance prédictive des ouvrages d’Art avec des fondations en site aquatiques, 11iemes Journées Nationales de Géotechnique et de Géologie de l’Ingénieur (JNGG), Lyon 2022')









