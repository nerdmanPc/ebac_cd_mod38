import pandas            as pd
import streamlit         as st
import numpy             as np

from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from pycaret.classification import *

def options_form():
    model_type, tune_model, plot_type = None, None, None
    with st.sidebar.form('options'):
        model_type = st.selectbox('Classifier type', [
            'lr',
            'lightgbm',
        ])
        tune_model = st.checkbox('Tune model', False)
        plot_type = st.selectbox('Plot type', [
            'auc',
            'ks',
            'pr',
            'feature',
            'confusion_matrix',
        ])
        st.form_submit_button('Apply')
    return {
        'model_type' : model_type, 
        'tune_model' : tune_model, 
        'plot_type' : plot_type,
    }

def init_data(data):
    data = pd.read_feather(data)
    data.dropna(inplace=True)
    data['safra'] = (data['data_ref'].dt.year - min(data['data_ref'].dt.year))*12 + data['data_ref'].dt.month
    data = data.reindex(columns=['renda'] + [col for col in data.columns if col != 'renda'])
    df_train = data[data.safra <= 12]
    df_test = data[data.safra > 12]
    return {
         'train': df_train,
         'test': df_test,
    }
     
def setup_pycaret(data):
    log_renda_trans = ColumnTransformer([('log_renda', FunctionTransformer(np.log), ['renda'])], remainder='passthrough')
    return setup(
        data=data, 
        target='mau',
        numeric_imputation='mean',
        ignore_features=['index', 'data_ref', 'safra'],
        max_encoding_ohe=-1,
        remove_outliers=True,
        outliers_method='lof',
        pca=True,
        pca_components=5,
        normalize=True,
        custom_pipeline=Pipeline([('log_renda', log_renda_trans)]),
        custom_pipeline_position=0,
        fold_strategy='timeseries',
        fold=5,
    )

def train_model(model_type):
    return create_model(model_type)

def calibrate_model(clf):
    return tune_model(clf, fold=3)

def evaluate_model(clf, plot_type):
    plot_model(clf, plot_type, display_format='streamlit')

def wrap_model(clf):
    return finalize_model(clf)

def main():

    st.set_page_config(page_title = 'Análise de Crédito', 
        initial_sidebar_state='expanded',
        layout="wide",
    )

    options = options_form()

    data = st.file_uploader('Upload file (.ftr).', type=['ftr'])

    if data is not None:
        data = init_data(data)
        setup_pycaret(data.train.sample(frac=0.1))
        clf = train_model(options.model_type)
        if options.tune_model:
            clf = calibrate_model(clf)
        evaluate_model(clf, options.plot_type)
        final_clf = wrap_model(clf)
        st.download_button('Download model', )
        st.button('Save model', on_click=save_model, args=(final_clf, f'final_credit_clf_ {date.today()}'))
        

if __name__ == '__main__':
	main()