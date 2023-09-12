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


def main():

    st.set_page_config(page_title = 'RFV', 
        initial_sidebar_state='expanded',
        layout="wide",
    )

    model_type, tune_model, plot_type = None, None, None
    with st.sidebar.form('options'):
        model_type = st.selectbox('Classifier type', [
            'lr',
            'lightgbm',
        ])
        tune_model = st.checkbox('Tune model', False)
        plot_type = st.selectbox('Plot type', [
            'confusion_matrix',
            'feature',
            'auc',
            'ks',
            'pr',
        ])
        st.form_submit_button('Apply')
         

    df = st.file_uploader('Upload file (.ftr).', type=['ftr'])

    if df is not None:
        df = pd.read_feather(df).dropna()

        df['safra'] = (df['data_ref'].dt.year - min(df['data_ref'].dt.year))*12 + df['data_ref'].dt.month
        df = df.reindex(columns=['renda'] + [col for col in df.columns if col != 'renda'])
        df_test = df[df.safra > 12]
        df_train = df[df.safra <= 12]

        log_renda_trans = ColumnTransformer([('log_renda', FunctionTransformer(np.log), ['renda'])], remainder='passthrough')

        exp = setup(
            data=df_train.sample(frac=0.1), 
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

        clf = create_model(model_type)

        tuned_clf = clf
        if tune_model:
            tuned_clf = tune_model(clf, fold=3)

        plot_model(tuned_clf, plot_type, display_format='streamlit')

        final_clf = finalize_model(tuned_clf)
        st.button('Save model', on_click=save_model, args=(final_clf, f'final_credit_clf_ {date.today()}'))
        

if __name__ == '__main__':
	main()