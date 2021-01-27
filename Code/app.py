import streamlit as st
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


def search_keywords(key_words, journals, year_begin, year_end, max_show, data_load_state):
    if button_clicked:
        data_load_state.markdown('Searching data...')
        if ' ' in key_words:
            if "\"" not in key_words:
                key_words_list = key_words.split(' ')
                key_words = ''.join([f'(?=.*{i})' for i in key_words_list])
        mask_jounral = df.journ.isin(journals)
        mask_year = (df.year >= year_begin) & (df.year <= year_end)
        dt = df.loc[mask_jounral & mask_year]
        mask_title = dt.title.str.contains(key_words, regex=True)
        mask_abstract = dt.abstract.str.contains(key_words, regex=True)
        dt = dt.loc[mask_title | mask_abstract]
        dt = dt.reset_index()
        data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
        show_papers(dt.head(max_show))
    else:
        data_load_state = data_load_state.markdown('**10 Random Papers**')
        dr = df.sample(10).reset_index()
        show_papers(dr)


if __name__ == '__main__':
    key_words = st.text_input('Keywords in Title and Abstract')
    journals = st.multiselect("Journals", ['aer', 'jpe', 'qje', 'ecta', 'res'], ['aer', 'jpe', 'qje', 'ecta', 'res'])
    c1, c2, c3 = st.beta_columns(3)
    year_begin = c1.number_input('Year Start', value=1980, min_value=1950, max_value=2020)
    year_end = c2.number_input('Year End', value=2020, min_value=1950, max_value=2021)
    max_show = c3.number_input('Max. Shown', value=100, min_value=0, max_value=500)
    button_clicked = st.button("Search")

    data_load_state = st.markdown('')

    df = load_data()

    search_keywords(key_words, journals, year_begin, year_end, max_show, data_load_state)
