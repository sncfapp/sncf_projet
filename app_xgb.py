import ast
import pickle
import sre_constants
import joblib
import streamlit
import pymongo
from mod import *
import pandas as pd
from PIL import Image
import time
import pymongo
import datetime


image = Image.open('../../data/2550815409b76ec968de461263ad7fe92b06789116ba8fcf7ea74a81.png')
st.image(image)
st.title('XGBoost classifier for bridge scour risk prediction')
st.text \
    ('This classfieir is trained and tested by using an SNCF \ndataset with the aim to predict the bridge pier scour risk.')

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
        oui = st.checkbox('OUI', value=False,key= "count")
        non = st.checkbox("NON", value=False,key= "count")

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
            ("Avez-vous des commentaires à ajouter ?")
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
    client = pymongo.MongoClient()
    client["sncf"]["formu_user"].insert(form_input)

st.header("References")
st.write('Wang, T., Reiffsteck, P., Chevalier, C., Zhu, Z., Chen, C.-W., & Schmidt, F. (2022). \n A novel extreme gradient boosting algorithm based model for predicting the scour risk around bridge piers: application to French railway bridges. European Journal of Environmental and Civil Engineering, 1–19.')
st.write("- [Article](https://doi.org/10.1080/19648189.2022.2072957)")
st.write('Wang, T., Reiffsteck, P., Chevalier, C., Chen, C.-W., Schmidt, F., Maintenance prédictive des ouvrages d’Art avec des fondations en site aquatiques, 11iemes Journées Nationales de Géotechnique et de Géologie de l’Ingénieur (JNGG), Lyon 2022')









