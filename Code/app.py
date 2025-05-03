import streamlit as st
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime



st.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='collapsed')

"""
# Econ Paper Search

"""

def load_single_file(file_path, args):
    """Load a single CSV file with memory optimization"""
    # Read CSV in chunks to reduce memory
    chunks = []
    for chunk in pd.read_csv(file_path, **args, chunksize=10000):
        # Convert object columns to categories where possible
        for col in chunk.select_dtypes(['object']).columns:
            if chunk[col].nunique() / len(chunk) < 0.5:  # If less than 50% unique values
                chunk[col] = chunk[col].astype('category')
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def load_data_and_combine():
    args = {
        "dtype": {
            "year": "Int16",  # Using smaller integer type
            "journal": "category"  # Store journal as category
        },
        "usecols": ["title", "authors", "abstract", "url", "journal", "year"]
    }

    # Load files one by one to prevent memory spike
    dfs = []
    file_periods = ['b2000', '2000s', '2010s', '2015s', '2020s']

    for period in file_periods:
        df = load_single_file(f"Data/papers_{period}.csv", args)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Clean data
    df = df[~df.year.isna()]

    # Use string dtype instead of object
    df['title'] = df['title'].astype('string')
    df['authors'] = df['authors'].astype('string')
    df['abstract'] = df['abstract'].astype('string')

    # drop book reviews (not perfect)
    masks = [~df.title.str.contains(i, case=False, regex=False) for i in ["pp.", " p."]]  # "pages," " pp "
    mask = np.vstack(masks).all(axis=0)
    df = df.loc[mask]
    # clean titles
    df.title = df.title.str.replace(r'\n\[.*?\]', '', regex=True)
    df.title = df.title.str.replace(r'\n', ' ', regex=True)
    # drop some duplicates due to weird strings in authors and abstract
    df = df[~df.duplicated(['title', 'url']) | df.url.isna()]
    # replace broken links to None
    broken_links = ["http://hdl.handle.net/", "https://hdl.handle.net/"]
    df.loc[df.url.isin(broken_links), "url"] = None

    return df


@st.cache_data  # Changed from st.cache
def load_data_cached(timestamp):
    return load_data_and_combine()


def load_data():
    update_timestamp = os.path.getmtime("Data/papers_2020s.csv")
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
            with st.expander("", expanded=True):
                st.markdown(row.abstract)


def search_keywords(
        button_clicked,
        df, data_load_state,
        key_words, journals, year_begin, year_end, sort_mth, max_show,
        show_abstract, search_author, random_roll):
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
        sort_map = {'Most recent': ['year', False], 'Most early': ['year', True], }  # 'Most cited': 'cite'
        # can use double sort: [sort_map[sort_mth], 'journal'], ascending=[False, True]
        dt = dt.sort_values(sort_map[sort_mth][0], ascending=sort_map[sort_mth][1]).reset_index()

        # show results
        if random_roll:
            data_load_state.markdown(f'**Roll From {dt.shape[0]} Papers**')
            show_papers(dt.sample(), show_abstract)
        else:
            data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
            show_papers(dt.head(int(max_show)), show_abstract)
    # else:
    #     data_load_state = data_load_state.markdown('**10 Random Papers**')
    #     dr = df.sample(10).reset_index()
    #     show_papers(dr)


def sidebar_info():
    st.sidebar.header("About")
    st.sidebar.markdown("""
    <div style="font-size: small; font-style: italic">
    This is a simple app to search for <b>economics papers</b> on leading economics journals.<br>
    It allows to <b>smart search</b> for only economics papers with selection of the set of economics jounrals.<br>
    The data is gathered from <b>RePEc</b> and will be <b>updated monthly (at 1st)</b>.<br>
    <br>
    <b>Author</b>: <a href="https://zhuxuanli.com" target="_blank" rel="noopener noreferrer">Xuanli Zhu</a><br>
    </div>
    """, unsafe_allow_html=True)

    update_timestamp = os.path.getmtime("Data/papers_2020s.csv")
    update_time_str = datetime.fromtimestamp(update_timestamp).strftime("%Y-%m-%d")
    st.sidebar.markdown(f"""
    <div style="font-size: small; font-style: italic;">
    Last data update: {update_time_str}<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Search Help")
    st.sidebar.markdown("""
    <div style="font-size: small; font-style: italic"">
    - The search looks for the papers with title and abstract that contain <b>all of the keywords</b> (split by space).<br>
    - The search does not distinguish between full words and <b>parts of words</b>.<br>
    - The search is <b>case insensitive</b>.<br>
    - The search allows for using double-quotes "" to find the <b>exact phrases</b> (with space inside).<br>
    - The search allows for using | between multiply words (no spaces) to match <b>either words</b>.<br>
    - The search will return all papers of the selected journals if the keywords are <b>blank</b>.<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Configs")
    full_journal = st.sidebar.checkbox("full set of journals", value=False)
    show_abstract = st.sidebar.checkbox("show abstract", value=False)
    search_author = st.sidebar.checkbox("search author", value=False)
    random_roll = st.sidebar.checkbox("random roll", value=False)

    st.sidebar.header("Journal Abbreviations")
    st.sidebar.markdown("""
    <div style="color: green; font-size: small">
    aejapp: AEJ Applied Economics<br>
    aejmac: AEJ Macroeconomics<br>
    aejmic: AEJ Micro<br>
    aejpol: AEJ Policy<br>
    aer: American Economic Review<br>
    aeri: AER Insights<br>
    are: Annual Review of Economics<br>
    ecta: Econometrica<br>
    ej: Economic Journal<br>
    eer: European Economic Review<br>
    ier: International Economic Review<br>
    jde: Journal of Development Economics<br>
    jeea: Journal of the European Economic Association<br>
    jeg: Journal of Economic Growth<br>
    jeh: The Journal of Economic History<br>
    jel: Journal of Economic Literature<br>
    jep: Journal of Economic Perspectives<br>
    jet: Journal of Economic Theory<br>
    jhe: Journal of Health Economics<br>
    jhr: Journal of Human Resources<br>
    jie: Journal of International Economics<br>
    joe: Journal of Econometrics<br>
    jole: Journal of Labor Economics<br>
    jme: Journal of Monetary Economics<br>
    jpe: Journal of Political Economy<br>
    jpemic: Journal of Political Economy Microeconomics<br>
    jpemac: Journal of Political Economy Macroeconomics<br>
    jpube: Journal of Public Economics<br>
    jue: Journal of Urban Economics<br>
    qe: Quantitative Economics<br>
    qje: Quarterly Journal of Economics<br>
    rand: RAND Journal of Economics<br>
    red: Review of Economic Dynamics<br>
    restud: Review of Economic Studies<br>
    restat: Review of Economics and Statistics<br>
    te: Theoretical Economics<br>
    <br>
    aerpp: AER Papers and Proceedings<br>
    ajhe: American Journal of Health Economics<br>
    aler: American Law and Economics Review<br>
    bpea: Brookings Papers on Economic Activity<br>
    cej: Canadian Journal of Economics<br>
    ecoa: Economica<br>
    ecot: Econometric Theory<br>
    er: Econometric Reviews<br>
    edcc: Economic Development and Cultural Change<br>
    ee: Experimental Economics<br>
    eedur: Economics of Education Review<br>
    eeh: Explorations in Economic History<br>
    efp: Education Finance and Policy<br>
    ehr: Economic History Review<br>
    ei: Economic Inquiry<br>
    geb: Games and Economic Behavior<br>
    ijio: International Journal of Industrial Organization<br>
    ijgt: International Journal of Game Theory<br>
    imfer: IMF Economic Review<br>
    jae: Journal of Applied Econometrics<br>
    jaere: Journal of the Association of Environmental and Resource Economists<br>
    jbf: Journal of Banking & Finance<br>
    jbes: Journal of Business & Economic Statistics<br>
    jdeme: Journal of Demographic Economics<br>
    jedc: Journal of Economic Dynamics and Control<br>
    jei: Journal of Economic Inequality<br>
    jemstr: Journal of Economics & Management Strategy<br>
    jeem: Journal of Environmental Economics and Management<br>
    jecsur: Journal of Economic Surveys<br>
    jebo: Journal of Economic Behavior & Organization<br>
    jf: Journal of Finance<br>
    jfe: Journal of Financial Economics<br>
    jhc: Journal of Housing Economics<br>
    jinde: Journal of Industrial Economics<br>
    jpope: Journal of Population Economics<br>
    jleo: Journal of Law, Economics, and Organization<br>
    jlawe: Journal of Law and Economics<br>
    jmcb: Journal of Money, Credit and Banking<br>
    jmate: Journal of Mathematical Economics<br>
    le: Labour Economics<br>
    md: Macroeconomic Dynamics<br>
    ms: Management Science<br>
    nberma: NBER Macroeconomics Annual<br>
    obes: Oxford Bulletin of Economics and Statistics<br>
    oxe: Oxford Economic Papers<br>
    qme: Quantitative Marketing and Economics<br>
    rsue: Regional Science and Urban Economics<br>
    rp: Research Policy<br>
    rfs: Review of Financial Studies<br>
    riw: Review of Income and Wealth<br>
    sje: Scandinavian Journal of Economics<br>
    smj: Strategic Management Journal<br>
    wber: World Bank Economic Review<br>
    <br>
    top5: [aer+ecta+jpe+qje+restud]<br>
    general: top5+[aeri+restat+jeea+eer+ej+qe]<br>
    survey: [jep+jel+are]<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Report Issues")
    st.sidebar.markdown("""
    <div style="font-size: small">
    Report an issue or comment at <a href="https://github.com/Alalalalaki/Econ-Paper-Search" target="_blank" rel="noopener noreferrer">github repo</a>
    </div>
    """, unsafe_allow_html=True)

    return show_abstract, search_author, random_roll, full_journal


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
    show_abstract, search_author, random_roll, full_journal = sidebar_info()
    # st.text(os.getcwd())
    hide_right_menu()

    local_css("Code/style.css")

    form = st.form(key='search')

    if not random_roll:
        key_words = form.text_input('Keywords in Title and Abstract')
        button_label = 'Search'
    else:
        key_words = ""
        button_label = 'Roll a rondom paper'

    a1, a2 = form.columns([1.08, 1]) # 1.53 without " & configs!"
    button_clicked = a1.form_submit_button(label=button_label)
    a2.markdown(
        """<div style="color: green; font-size: small; padding-bottom: 0;">
        (see left sidebar for search help & journal abbrevs & <font color="blue">configs</font>!)
        </div>""",
        unsafe_allow_html=True)

    # alternatively add show abstract here
    # show_abstract = a3.checkbox("show abstract", value=False)

    js = ['aer', 'jpe', 'qje', 'ecta', 'restud',
          'aejmac', 'aejmic', 'aejapp', 'aejpol', 'aeri', 'jpemic', 'jpemac',
          'restat', 'jeea', 'eer', 'ej',
          'jep', 'jel', 'are',
          'qe', 'jeg',
          'jet', 'te', 'joe',
          'jme', 'red', 'rand', 'jole', 'jhr',
          'jie', 'ier', 'jpube', 'jde',
          'jeh',"jue",
          "jhe",
          ]
    js_ = ["jae","geb","jinde","jlawe","jebo","ee","ectt",'ehr','eeh',"imfer","ecot","jmcb","edcc","sje","ecoa",
            "jaere","jeem","wber","ijio","jleo","le","jpope","qme","ei","jedc","cej","obes","jems","jes","jmate",
            "rsue","eedur","jhc","efp","aler","jbes",
            "jf","jfe","rfs","ms","jbf","smj","rp","bpea","er","ijgt","ntj","md","jdeme","oxe","jei","riw","ajhe",
            "nberma","aerpp",
            ]
    if full_journal:
        js += js_
    js_cats = {"all": js,
               "top5": ['aer', 'jpe', 'qje', 'ecta', 'restud'],
               "general": ['aer', 'jpe', 'qje', 'ecta', 'restud', 'aeri', 'restat', 'jeea', 'eer', 'ej', 'qe'],
               "survey": ['jep', 'jel', 'are', ]
               }
    js_cats_keys = list(js_cats.keys())
    journals = form.multiselect("Journals",
                                js_cats_keys+js, js)  # js[:21] //
    # if selected journals include js_cats
    js_temp = set(journals) & set(js_cats_keys)
    if js_temp:
        for c in js_temp:
            journals += js_cats[c]
        journals = set(journals)

    year_min = 1900
    year_max = datetime.now().year

    c1, c2, c3, c4 = form.columns(4)
    year_begin = c1.number_input('Year from', value=1980, min_value=year_min, max_value=year_max)
    year_end = c2.number_input('Year to', value=year_max, min_value=year_min, max_value=year_max)
    sort_mth = c3.selectbox('Sort by', ['Most recent', 'Most early'], index=0)  # 'Most cited'
    max_show = c4.number_input('Max. Shown', value=100, min_value=0, max_value=500)

    data_load_state = st.empty()

    df = load_data()

    search_keywords(button_clicked,
                    df, data_load_state,
                    key_words, journals, year_begin, year_end, sort_mth, max_show,
                    show_abstract, search_author, random_roll)


if __name__ == '__main__':
    main()
