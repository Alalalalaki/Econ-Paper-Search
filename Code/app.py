import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='collapsed')

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


def search_keywords(key_words, journals, year_begin, year_end, sort_mth, max_show, data_load_state):
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
        sort_map = {'Most recent': 'year', 'Most cited': 'cite'}
        dt = dt.sort_values(sort_map[sort_mth], ascending=False).reset_index()
        data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
        show_papers(dt.head(max_show))
    # else:
    #     data_load_state = data_load_state.markdown('**10 Random Papers**')
    #     dr = df.sample(10).reset_index()
    #     show_papers(dr)


def sidebar_info():
    st.sidebar.header("About")
    st.sidebar.markdown("This is a simple app to search for economic papers by economic journals.")
    st.sidebar.header("Help")
    st.sidebar.subheader("Search Help")
    st.sidebar.markdown("""
    <div style="font-size: small">
    - The search looks for the papers with title and abstract that contain all of the keywords.<br>
    - The search does not distinguish between full words and parts of words.<br>
    - The search is case insensitive.<br>
    - Use double-quotes to find exact phrases.<br>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.subheader("Journal Abbreviations")
    st.sidebar.markdown("""
    <div style="color: green; font-size: small">
    aejmic: AEJ Micro<br>
    aejpol: AEJ Policy<br>
    aejapp: AEJ Applied Economics<br>
    aejmac: AEJ Macroeconomics<br>
    aer: American Economic Review<br>
    aeri: AER Insights<br>
    ecta: Econometrica<br>
    jeea: Journal of the European Economic Association<br>
    jep: Journal of Economic Perspectives<br>
    jpe: Journal of Political Economy<br>
    pandp: AEA Papers and Proceedings<br>
    qje: Quarterly Journal of Economics<br>
    restud: Review of Economic Studies<br>
    restat: Review of Economics and Statistics<br>
    </div>
    """, unsafe_allow_html=True)


def main():
    sidebar_info()

    key_words = st.text_input('Keywords in Title and Abstract')
    js = ['aer', 'jpe', 'qje', 'ecta', 'restud', 'aejmac', 'aejmic', 'aejapp', 'aejpol', 'aeri', 'restat', 'jeea', 'jep']
    journals = st.multiselect("Journals", js, js)
    c1, c2, c3, c4 = st.beta_columns(4)
    year_begin = c1.number_input('Year from', value=1980, min_value=1950, max_value=2020)
    year_end = c2.number_input('Year to', value=2020, min_value=1950, max_value=2021)
    sort_mth = c3.selectbox('Sort by', ['Most recent', 'Most cited'], index=0)
    max_show = c4.number_input('Max. Shown', value=100, min_value=0, max_value=500)
    button_clicked = st.button("Search")

    data_load_state = st.empty()

    df = load_data()

    search_keywords(key_words, journals, year_begin, year_end, sort_mth, max_show, data_load_state)

if __name__ == '__main__':
    main()

