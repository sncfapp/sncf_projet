import time
import streamlit as st
import os
import pandas as pd
from mod import *
from sklearn.model_selection import train_test_split

def model_selector():
    model_training_container = st.sidebar.expander("Load a model", True)
    with model_training_container:
        path = "/Users/caps/Documents/prestations/sncf/test/scour_risk/model_d/"
        model_name = st.selectbox(
            "Choose a model",
            os.listdir(path),
        )
        dataset = st.selectbox("Choose a dataset",
                               os.listdir("/Users/caps/Documents/prestations/sncf/test/scour_risk/data_d"))

    import joblib
    return joblib.load(path+model_name), dataset


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data(data_name):

    dforiginal = pd.read_excel(f'/Users/caps/Documents/prestations/sncf/test/scour_risk/data_d/{data_name}', engine='openpyxl')
    y = dforiginal["Risk_bin"]
    df_X = dforiginal.drop(columns=['Risk_bin'])
    dummy = pd.get_dummies(df_X["Fdt_type"], prefix="Fdt_type")
    df = pd.concat([df_X, dummy], axis=1)
    del df["Fdt_type"]
    col = ['Flow_type', 'Slope_bed(%)', 'Flow', 'WV/WC', 'Topo', 'Sinuosity', 'Riv_material', 'Scpt_scour',
           'Flood_hstry', 'Scour_hstry', 'Channel_rating', 'Bank_rating', 'Pier_shape', 'Ctms_foundation', 'Masonry_YN',
           'Local_scour_YN', 'Other_damages', 'Fdt_type_Caisson', 'Fdt_type_Concrete/ciment', 'Fdt_type_Timber piles']

    df.columns = col
    return train_test_split(df, y)
#
def sidebar_controllers():
    model, dataset = model_selector()
    X_train, X_test, y_train, y_test = generate_data(
        dataset)


    return (
        dataset,
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

st.header('Admin Dashboard')

etat = st.sidebar.selectbox('',["train", "explore", "predict"])

if etat == "train":
    model_selector()



if etat == "explore":
    st.info('Visualize and download dataset')
    df_form = form_data()
    df_hist = model_historique()
    st.header('Form Values')
    st.dataframe(df_form)
    st.header('Historique predictions')
    download_data()

    b = st.checkbox('Only errors prediction')
    if b:
        st.dataframe(df_hist[df_hist["avis_user"]=="neg"])
    else:
        st.dataframe(df_hist)


if etat == "predict":
    st.info('select the features and start prediction')
    st.session_state["user_data"], _ = user_input_features()
    st.session_state["row"], st.session_state['y'] = pipeline(st.session_state["user_data"])
    if st.button("run model"):
        classe, proba, xgb = pred(st.session_state["row"])
        gauge(round(proba[classe] * 100, 2))
        time.sleep(0.5)
        st.success(f"class prediction {classe}")
        plot1(xgb)
        plot2(xgb)
        plot3(xgb)
        plot4(xgb)



if "model" and "dataset" in st.session_state:
    st.button('run train')





