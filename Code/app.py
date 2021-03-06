import streamlit as st
import numpy as np
import pandas as pd
import os
import re


st.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='collapsed')

"""
# Econ Paper Search

"""


@st.cache(show_spinner=False)
def load_data_cached(timestamp):
    df = pd.read_csv("Data/papers.csv",
                     usecols=["title", "authors", "abstract", "url", "jel", "journal", "year"],
                     dtype={"year": "Int16"}
                     ).drop_duplicates()
    df = df[~df.year.isna()]
    # drop book reviews (not perfect)
    masks = [~df.title.str.contains(i, case=False, regex=False) for i in ["pp.", " p."]]  # "pages," " pp "
    mask = np.vstack(masks).all(axis=0)
    df = df.loc[mask]
    # drop some duplicates due to weird strings in authors and abstract
    df = df[~df.duplicated(['title', 'url']) | df.url.isna()]
    # replace broken links to None
    broken_links = ["http://hdl.handle.net/", ]
    df.loc[df.url.isin(broken_links), "url"] = None
    return df


def load_data():
    update_timestamp = os.path.getmtime("Data/papers.csv")
    return load_data_cached(update_timestamp)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def show_papers(df, show_abstract):
    for index, row in df.iterrows():
        if row.url:
            st.markdown(f'{index+1}.  [{row.title}]({row.url}). {row.authors}. {row.year}. {row.journal}.')
        else:
            st.markdown(f'{index+1}.  {row.title}. {row.authors}. {row.year}. {row.journal}.')
        if show_abstract:
            with st.beta_expander(""):
                st.markdown(row.abstract)


def search_keywords(
        button_clicked,
        df, data_load_state,
        key_words, journals, year_begin, year_end, sort_mth, max_show,
        show_abstract, search_author):
    if button_clicked:
        data_load_state.markdown('Searching paper...')

        # preliminary select on
        mask_jounral = df.journal.isin(journals)
        mask_year = (df.year >= year_begin) & (df.year <= year_end)
        dt = df.loc[mask_jounral & mask_year]
        info = dt.title + ' ' + dt.abstract.fillna('')
        if search_author:
            info = info + ' ' + dt.authors

        # the case of ""
        if (' ' in key_words) & ("\"" not in key_words):
            # the case of \s but no ""
            key_words = key_words.split(' ')
            # key_words = ''.join([f'(?=.*{i})' for i in key_words_list])
        else:
            if "\"" in key_words:
                # the case of ""
                _key_words = re.findall(r'(?:"[^"]*"|[^\s"])+', key_words)
                key_words = [i.replace("\"", "") for i in _key_words]
            else:
                # the case of no \s and no ""
                key_words = [key_words]

        # the case of |
        key_words_or = [s for s in key_words if "|" in s]
        if key_words_or:
            mask_or = []
            for kws in key_words_or:
                kws = kws.split("|")
                masks_or = [info.str.contains(s, case=False, regex=False) for s in kws]
                mask_or.append(np.vstack(masks_or).any(axis=0))
            mask_or = np.vstack(mask_or).all(axis=0)
            key_words = [s for s in key_words if s not in key_words_or]
            if not key_words:
                key_words = [""]
        else:
            mask_or = [True]*len(dt)

        # final select on
        masks = [info.str.contains(s, case=False, regex=False) for s in key_words]
        mask = np.vstack([np.vstack(masks), mask_or]).all(axis=0)
        dt = dt.loc[mask]

        # sort
        sort_map = {'Most recent': 'year', 'Most cited': 'cite'}
        # can use double sort: [sort_map[sort_mth], 'journal'], ascending=[False, True]
        dt = dt.sort_values(sort_map[sort_mth], ascending=False).reset_index()

        # show results
        data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
        show_papers(dt.head(max_show), show_abstract)
    # else:
    #     data_load_state = data_load_state.markdown('**10 Random Papers**')
    #     dr = df.sample(10).reset_index()
    #     show_papers(dr)


def sidebar_info():
    st.sidebar.header("About")
    st.sidebar.markdown("""
    <div style="font-size: small; font-style: italic">
    This is a simple app to search for economic papers by economic journals.<br>
    The data is gathered from RePEc and will be updated monthly.<br>
    </div>
    """, unsafe_allow_html=True)  # Author: Xuanli Zhu.<br>

    st.sidebar.header("Search Help")
    st.sidebar.markdown("""
    <div style="font-size: small">
    - The search looks for the papers with title and abstract that contain all of the keywords.<br>
    - The search does not distinguish between full words and parts of words.<br>
    - The search is case insensitive.<br>
    - The search allows for using double-quotes "" to find the exact phrases.<br>
    - The search allows for using | between multiply words (no spaces) to match either words.<br>
    - The search will return all papers of the selected journals if the keywords are blank.<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Journal Abbreviations")
    st.sidebar.markdown("""
    <div style="color: green; font-size: small">
    aejmic: AEJ Micro<br>
    aejpol: AEJ Policy<br>
    aejapp: AEJ Applied Economics<br>
    aejmac: AEJ Macroeconomics<br>
    aer: American Economic Review<br>
    aeri: AER Insights<br>
    are: Annual Review of Economics<br>
    ecta: Econometrica<br>
    ehr: Economic History Review<br>
    ej: Economic Journal<br>
    eer: European Economic Review<br>
    eeh: Explorations in Economic History<br>
    jde: Journal of Development Economics<br>
    jeea: Journal of the European Economic Association<br>
    jel: The Journal of Economic History<br>
    jel: Journal of Economic Literature<br>
    jep: Journal of Economic Perspectives<br>
    jet: Journal of Economic Theory<br>
    jie: Journal of International Economics<br>
    joe: Journal of Econometrics<br>
    jole: Journal of Labor Economics<br>
    jme: Journal of Monetary Economics<br>
    jpe: Journal of Political Economy<br>
    jpube: Journal of Public Economics<br>
    qe: Quantitative Economics<br>
    qje: Quarterly Journal of Economics<br>
    rand: RAND Journal of Economics<br>
    red: Review of Economic Dynamics<br>
    restud: Review of Economic Studies<br>
    restat: Review of Economics and Statistics<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Experimental Config")
    show_abstract = st.sidebar.checkbox("show abstract", value=False)
    search_author = st.sidebar.checkbox("search author", value=False)

    st.sidebar.header("Report Issues")
    st.sidebar.markdown("""
    <div style="font-size: small">
    Report an issue or comment at <a href="https://github.com/Alalalalaki/Econ-Paper-Search">github repo</a>
    </div>
    """, unsafe_allow_html=True)

    return show_abstract, search_author


def hide_right_menu():
    # ref: https://discuss.streamlit.io/t/how-do-i-hide-remove-the-menu-in-production/362/3
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    show_abstract, search_author = sidebar_info()
    # st.text(os.getcwd())
    hide_right_menu()

    local_css("Code/style.css")

    key_words = st.text_input('Keywords in Title and Abstract')
    button_clicked = st.button("Search")

    js = ['aer', 'jpe', 'qje', 'ecta', 'restud',
          'aejmac', 'aejmic', 'aejapp', 'aejpol', 'aeri',
          'restat', 'jeea', 'eer', 'ej',
          'jep', 'jel', 'are',
          'qe',
          'jet', 'joe',
          'jme', 'red', 'rand', 'jole',
          'jie', 'jpube', 'jde',
          'jeh', 'ehr', 'eeh',
          ]
    journals = st.multiselect("Journals", js, js)  # js[:21] // (see left sidebar for journal abbreviations)

    year_min = 1900
    year_max = 2021

    c1, c2, c3, c4 = st.beta_columns(4)
    year_begin = c1.number_input('Year from', value=1980, min_value=year_min, max_value=year_max)
    year_end = c2.number_input('Year to', value=year_max, min_value=year_min, max_value=year_max)
    sort_mth = c3.selectbox('Sort by', ['Most recent', ], index=0)  # 'Most cited'
    max_show = c4.number_input('Max. Shown', value=100, min_value=0, max_value=500)

    data_load_state = st.empty()

    df = load_data()

    search_keywords(button_clicked,
                    df, data_load_state,
                    key_words, journals, year_begin, year_end, sort_mth, max_show,
                    show_abstract, search_author)


if __name__ == '__main__':
    main()
