import streamlit as st
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from data_processing import load_all_papers  # Use shared data loading

import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

st.set_page_config(page_title="Econ Paper Search", page_icon=None, layout='centered', initial_sidebar_state='auto')

"""
# Econ Paper Search

"""


@st.cache_data
def load_data_cached(timestamp):
    """Load data using the shared processing module"""
    df = load_all_papers()
    return df


def load_data():
    """Load data with caching based on file modification time"""
    update_timestamp = os.path.getmtime("Data/papers_2020s.csv")
    return load_data_cached(update_timestamp)


@st.cache_data
def load_embeddings_cached(timestamp):
    """Load embeddings in the exact same order as papers were processed during generation."""
    all_embeddings = []

    for period in ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s']:
        path = f'Embeddings/embeddings_{period}.npy'
        if os.path.exists(path):
            embeddings = np.load(path).astype(np.float32)
            all_embeddings.append(embeddings)

    # Concatenate all embeddings in order
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return None


def load_embeddings():
    """Load embeddings with caching based on file modification time"""
    update_timestamp = os.path.getmtime("Embeddings/embeddings_2020s.npy")
    return load_embeddings_cached(update_timestamp)


def lazy_load_semantic_search():
    """Lazy load semantic search module only when needed"""
    from semantic_search import perform_semantic_search, load_semantic_model
    return perform_semantic_search, load_semantic_model


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
        mask_journal = df.journal.isin(journals)
        mask_year = (df.year >= year_begin) & (df.year <= year_end)
        dt = df.loc[mask_journal & mask_year]
        info = dt.title + ' ' + dt.abstract.fillna('')
        if search_author:
            info = info + ' ' + dt.authors

        # the case of ""
        if (' ' in key_words) & ("\"" not in key_words):
            # the case of \s but no ""
            key_words = key_words.split(' ')
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
        sort_map = {'Most recent': ['year', False], 'Most early': ['year', True], }
        dt = dt.sort_values(sort_map[sort_mth][0], ascending=sort_map[sort_mth][1]).reset_index(drop=True)

        # show results
        if random_roll:
            data_load_state.markdown(f'**Roll From {dt.shape[0]} Papers**')
            show_papers(dt.sample(), show_abstract)
        else:
            data_load_state.markdown(f'**{dt.shape[0]} Papers Found**')
            show_papers(dt.head(int(max_show)), show_abstract)


def search_semantic(
        button_clicked,
        df, data_load_state,
        query, journals, year_begin, year_end, sort_mth, min_similarity, max_show,
        show_abstract, random_roll):
    """Handle semantic search"""
    if button_clicked:
        if not query.strip() and not random_roll:
            data_load_state.markdown('Please enter a search query for AI search.')
            return

        data_load_state.markdown('Searching papers using AI...')

        # Pre-filter by journal and year
        mask_journal = df.journal.isin(journals)
        mask_year = (df.year >= year_begin) & (df.year <= year_end)
        mask = mask_journal & mask_year

        # Filter dataframe
        filtered_df = df[mask].copy()

        if len(filtered_df) == 0:
            data_load_state.markdown('**No papers found in selected journals/years**')
            return

        # Handle random roll
        if random_roll:
            data_load_state.markdown(f'**Roll From {filtered_df.shape[0]} Papers**')
            show_papers(filtered_df.sample(1), show_abstract)
            return

        # Now load embeddings and model only when actually needed
        try:

            # Load embeddings
            with st.spinner('Loading embeddings...'):
                embeddings = load_embeddings()
                if embeddings is None:
                    st.error("Failed to load embeddings!")
                    return

                # Check for dimension mismatch
                if len(embeddings) != len(df):
                    st.error(f"⚠️ Data mismatch detected!")
                    st.error(f"Papers in database: {len(df):,}")
                    st.error(f"Embeddings loaded: {len(embeddings):,}")
                    st.error("Please regenerate embeddings by running: python generate_embeddings.py")
                    return

                # Filter embeddings using the same mask
                filtered_embeddings = embeddings[mask]

            # Lazy load semantic search functions
            with st.spinner('Loading AI model...'):
                perform_semantic_search, load_semantic_model = lazy_load_semantic_search()

            # Perform semantic search with filtered data and embeddings
            results = perform_semantic_search(query, filtered_df, filtered_embeddings, min_similarity)

            if len(results) == 0:
                data_load_state.markdown(f'**No papers found with similarity ≥ {min_similarity:.2f}**')
                data_load_state.markdown('💭 Try lowering the similarity threshold or enriching your query terms.')
                return

            # Sort results based on user preference
            if sort_mth == 'Most recent':
                results = results.sort_values(['year', 'similarity'], ascending=[False, False])
            elif sort_mth == 'Most early':
                results = results.sort_values(['year', 'similarity'], ascending=[True, False])
            else:  # Best match
                results = results.sort_values('similarity', ascending=False)

            # Reset index to ensure correct numbering
            results = results.reset_index(drop=True)

            # Get total results before limiting
            total_results = len(results)

            # Limit to max_show
            results = results.head(int(max_show))

            # Show results with similarity scores
            data_load_state.markdown(f'**{total_results} Papers Found** (similarity ≥ {min_similarity:.2f})')

            # Custom display with similarity scores
            for idx, (_, row) in enumerate(results.iterrows()):
                similarity_badge = f'<span style="background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{row.similarity:.3f}</span>'

                if row.url:
                    st.markdown(f'{idx+1}. {similarity_badge} [{row.title}]({row.url}). {row.authors}. {row.year}. {row.journal}.', unsafe_allow_html=True)
                else:
                    st.markdown(f'{idx+1}. {similarity_badge} {row.title}. {row.authors}. {row.year}. {row.journal}.', unsafe_allow_html=True)

                if show_abstract:
                    with st.expander("", expanded=True):
                        st.markdown(row.abstract)

        except Exception as e:
            data_load_state.markdown(f'**Error in AI search: {str(e)}**')
            st.error("Please make sure embeddings are generated. Run generate_embeddings.py in the Code directory.")


def sidebar_info():
    st.sidebar.header("About")
    st.sidebar.markdown("""
    <div style="font-size: small;">
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
    Last update: {update_time_str}<br>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Configs")

    # Create columns for search mode and min similarity
    col1, col2 = st.sidebar.columns([1, 1])

    with col1:
        search_mode = st.radio("Search Mode", ["Keyword", "AI"], index=0, horizontal=True,
                              help="AI mode is powered by a small language embedding model (all-MiniLM-L6-v2) and uses semantic similarity to find related papers")

    # Min similarity number input (only for AI mode)
    min_similarity = 0.5  # default
    if search_mode == "AI":
        with col2:
            min_similarity = st.number_input('Min Sim',
                                           min_value=0.0,
                                           max_value=1.0,
                                           value=0.5,
                                           step=0.05,
                                           format="%.2f",
                                           help="Minimum similarity threshold")

    full_journal = st.sidebar.checkbox("full set of journals", value=False)
    show_abstract = st.sidebar.checkbox("show abstract", value=False)
    search_author = st.sidebar.checkbox("search author", value=False,
                                       disabled=(search_mode == "AI"),
                                       help="Author search is only available in Keyword mode")
    random_roll = st.sidebar.checkbox("random roll", value=False,
                                     disabled=(search_mode == "AI"),
                                     help="Random roll is only available in Keyword mode")

    # Conditional Search Help based on mode
    if search_mode == "Keyword":
        st.sidebar.header("Search Help (Keyword)")
        st.sidebar.markdown("""
        <div style="font-size: small;">
        - The search looks for the papers with title and abstract that contain <b>all of the keywords</b> (split by space).<br>
        - The search does not distinguish between full words and <b>parts of words</b>.<br>
        - The search is <b>case insensitive</b>.<br>
        - The search allows for using double-quotes "" to find the <b>exact phrases</b> (with space inside).<br>
        - The search allows for using | between multiply words (no spaces) to match <b>either words</b>.<br>
        - The search will return all papers of the selected journals if the keywords are <b>blank</b>.<br>
        </div>
        """, unsafe_allow_html=True)
    else:  # AI mode
        st.sidebar.header("Search Help (AI)")
        st.sidebar.markdown("""
        <div style="font-size: small;">
        - Use <b>natural language queries</b> to find semantically related papers.<br>
        - <b>Full phrases and sentences</b> work better than individual keywords.<br>
        - <b>Word order</b> matters; the same words in different contexts carry different meanings.<br>
        - <b>Examples</b>: "impact of minimum wage on employment in asian countries" or "impact of firm wage premium on wage inequality".<br>
        - The <b>similarity threshold</b> filters results: 0.5 is a rule of thumb.<br>
        </div>
        """, unsafe_allow_html=True)

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

    return show_abstract, search_author, random_roll, full_journal, search_mode, min_similarity


def apply_custom_css():
    """Apply custom CSS including optional menu hiding"""
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def main():
    apply_custom_css()
    show_abstract, search_author, random_roll, full_journal, search_mode, min_similarity = sidebar_info()

    local_css("Code/style.css")

    form = st.form(key='search')

    if not random_roll:
        key_words = form.text_input('Search Query')
        button_label = 'Search'
    else:
        key_words = ""
        button_label = 'Roll a random paper'

    a1, a2 = form.columns([1.18, 1])
    button_clicked = a1.form_submit_button(label=button_label)
    a2.markdown(
        """<div style="color: green; font-size: small; padding-bottom: 0;">
        (see left sidebar for search help & <font color="blue">new config of AI search</font>!)
        </div>""",
        unsafe_allow_html=True)

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
                                js_cats_keys+js, js)
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

    # Sort option for both modes
    if search_mode == "AI":
        sort_options = ['Most recent', 'Most early', 'Best match']
        sort_mth = c3.selectbox('Sort by', sort_options, index=0)
    else:
        sort_mth = c3.selectbox('Sort by', ['Most recent', 'Most early'], index=0)

    max_show = c4.number_input('Max. Shown', value=100, min_value=0, max_value=500)

    # Load data
    df = load_data()

    data_load_state = st.empty()

    # Call appropriate search function based on mode
    if search_mode == "AI":
        search_semantic(button_clicked,
                       df, data_load_state,
                       key_words, journals, year_begin, year_end, sort_mth, min_similarity, max_show,
                       show_abstract, random_roll)
    else:
        search_keywords(button_clicked,
                       df, data_load_state,
                       key_words, journals, year_begin, year_end, sort_mth, max_show,
                       show_abstract, search_author, random_roll)


if __name__ == '__main__':
    main()
