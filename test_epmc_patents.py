import urllib.request
import json
import urllib.parse

def search_epmc_patents(query):
    # epmc query for patents
    epmc_query = f'(SRC:PAT) AND ({query})'
    encoded_query = urllib.parse.quote(epmc_query)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&resultType=core"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get('resultList', {}).get('result', [])
            for res in results[:3]:
                print("ID:", res.get('id'))
                print("Title:", res.get('title'))
                print("Assignee:", res.get('authorString') or res.get('assignee'))
                print("Abstract:", res.get('abstractText')[:100] if res.get('abstractText') else 'None')
                print("---")
    except Exception as e:
        print("Error:", e)

search_epmc_patents("machine learning")
