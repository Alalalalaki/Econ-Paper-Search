import streamlit as st
import numpy as np
import pandas as pd


"""
# Econ Paper Search

"""


@st.cache
def load_data():
    df = pd.read_csv("../Data/article.csv")
    return df


def show_papers(df):
    for index, row in df.iterrows():
        st.markdown(f'{index+1}.  [{row.title}]({row.article_url}). {row.journ}. {row.year}.')


# @st.cache
def search_keywords(key_words, journals, year_begin, year_end, max_show, data_load_state):
    if button_clicked:
        data_load_state.markdown('Searching paper...')
        if (' ' in key_words) & ("\"" not in key_words):
            key_words = key_words.split(' ')
            # key_words = ''.join([f'(?=.*{i})' for i in key_words_list])
        else:
            key_words = [key_words]
        mask_jounral = df.journ.isin(journals)
        mask_year = (df.year >= year_begin) & (df.year <= year_end)
        dt = df.loc[mask_jounral & mask_year]
        info = dt.title + ' ' + dt.abstract.fillna('')
        masks = [info.str.contains(s, case=False) for s in key_words]
        mask = np.vstack(masks).all(axis=0)
        dt = dt.loc[mask]
        dt = dt.reset_index()
        data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
        show_papers(dt.head(max_show))
    # else:
    #     data_load_state = data_load_state.markdown('**10 Random Papers**')
    #     dr = df.sample(10).reset_index()
    #     show_papers(dr)


if __name__ == '__main__':
    key_words = st.text_input('Keywords in Title and Abstract')
    js = ['aer', 'jpe', 'qje', 'ecta', 'restud', 'aejmac', 'aejmic', 'aejapp', 'aejmac', 'aeri', 'restat', 'jeea', 'jep']
    journals = st.multiselect("Journals", js, js)
    c1, c2, c3 = st.beta_columns(3)
    year_begin = c1.number_input('Year Start', value=1980, min_value=1950, max_value=2020)
    year_end = c2.number_input('Year End', value=2020, min_value=1950, max_value=2021)
    max_show = c3.number_input('Max. Shown', value=100, min_value=0, max_value=500)
    button_clicked = st.button("Search")

    data_load_state = st.markdown('')

    df = load_data()

    search_keywords(key_words, journals, year_begin, year_end, max_show, data_load_state)
