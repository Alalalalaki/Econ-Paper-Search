import pandas as pd
from pandas.core.frame import DataFrame
from requests_html import HTMLSession
import re
from tqdm import tqdm
import time
from tenacity import retry, stop_after_attempt, wait_random

output_path = "../Data/"


def scraping_list(session, code):
    base_url = f"https://ideas.repec.org/s/{code}.html"

    r = session.get(base_url).html
    paper_links = [i for d in r.find("#content>.panel-body") for i in d.absolute_links]
    page = r.find(".pagination", first=True)
    if not page:
        return paper_links
    else:
        page_n = int(page.find("li")[-2].text)

    @retry(stop=stop_after_attempt(2), wait=wait_random(min=1, max=2))
    def get_links(url):
        r = session.get(url).html
        paper_links = [i for d in r.find("#content>.panel-body") for i in d.absolute_links]
        return paper_links

    for n in tqdm(range(2, page_n+1)):
        time.sleep(0.5)
        url = base_url.replace(".html", f"{n}.html")
        paper_links += get_links(url)
    return paper_links


def parse_paper(r):
    d = {}
    d["title"] = r.find("#title", first=True).text
    d["authors"] = r.find("#listed-authors", first=True).text.replace("Listed:\n", "")
    d["abstract"] = r.find("#abstract-body", first=True).text
    biblio = r.find("#biblio-body", first=True).text
    handle = re.search("RePEc:(?P<repec>[^\s]+)", biblio)
    d["bib_handle"] = handle.group("repec") if handle else None
    note = re.search("DOI: (?P<doi>[^\s]+)", biblio)
    d["bib_note"] = note.group("doi") if note else None
    download = r.find("#download", first=True).text
    url = re.search("(?P<url>https?://[^\s]+)", download)
    d["url"] = url.group("url") if url else None
    jel = r.find("#more>ul", first=True)
    d["jel"] = "&".join([re.search("(?P<jel>[A-Z]\d*)", i).group("jel") for i in jel.links]) if jel else None
    return d


def scraping_paper(session, paper_links):
    paper_info = []

    @retry(stop=stop_after_attempt(4), wait=wait_random(min=1, max=2))
    def get_info(url):
        r = session.get(url).html
        info = parse_paper(r)
        return info

    for url in tqdm(paper_links):
        time.sleep(0.5)
        try:
            info = get_info(url)
        except:
            print(url)
        paper_info.append(info)
    return paper_info


def main():
    journals_map = {
        "aea/aecrev": "aer",
        "ucp/jpolec": 'jpe',
        "oup/qjecon": 'qje',
        "ecm/emetrp": 'ecta',  # old
        "wly/emetrp": 'ecta',
        "oup/restud": 'restud',
        "aea/aejmac": 'aejmac',
        "aea/aejmic": 'aejmic',
        "aea/aejpol": 'aejpol',
        "aea/aejapp": 'aejapp',
        "aea/aerins": 'aeri',
        "aea/jeclit": 'jel',
        "aea/jecper": 'jep',
        "tpr/restat": 'restat',
        "tpr/jeurec": 'jeea',  # old
        "bla/jeurec": 'jeea',  # old
        "oup/jeurec": 'jeea',
        "eee/eecrev": 'eer',
        "ecj/econjl": 'ej',  # old
        "wly/econjl": 'ej',  # old
        "oup/econjl": 'ej',
        # field
        "eee/moneco": 'jme',
        "red/issued": 'red',
        "rje/randje": 'rand',  # old
        "bla/randje": 'rand',
        "ucp/jlabec": 'jole',
        "eee/inecon": 'jie',
        "eee/pubeco": 'jpube',
        "eee/deveco": 'jde',
        "cup/jechis": 'jeh',
        "bla/ehsrev": 'ehr',
        "eee/exehis": 'eeh',
    }
    session = HTMLSession()
    df = pd.DataFrame()
    for key, value in journals_map.items():
        print(f"{value}:")
        paper_links = scraping_list(session, key)
        paper_info = scraping_paper(session, paper_links)
        _df = pd.DataFrame(paper_info).assign(journal=value)
        df = pd.concat([df, _df], ignore_index=True)
    # df = df.assign(year=df.bib_handle.str.extract("y:(\d+)"))
    df.to_pickle(output_path+"papers.pkl")


if __name__ == '__main__':
    main()
