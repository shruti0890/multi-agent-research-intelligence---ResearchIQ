import urllib.request
import urllib.parse
import json

# 1. Test USPTO Open Data Portal
try:
    uspto_url = "https://developer.uspto.gov/ibd-api/v1/application/publications?searchText=multi-agent%20artificial%20intelligence&rows=5"
    req = urllib.request.Request(uspto_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        print(f"USPTO IBD Status: {resp.status}, count: {data.get('response', {}).get('numFound', 0)}")
        docs = data.get("response", {}).get("docs", [])
        for d in docs:
            print(f"  USPTO -> {d.get('inventionTitle')}")
except Exception as e:
    print(f"USPTO IBD Error: {e}")

# 2. Test USPTO PPUBS search endpoint
try:
    ppubs_url = "https://ppubs.uspto.gov/dirsearch-public/patents/search"
    payload = json.dumps({
        "searchText": '("multi-agent" OR "multiagent") AND ("artificial intelligence" OR "machine learning")',
        "q": '("multi-agent" OR "multiagent") AND ("artificial intelligence" OR "machine learning")',
        "start": 0,
        "limit": 5
    }).encode()
    req = urllib.request.Request(ppubs_url, data=payload, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        print(f"PPUBS Status: {resp.status}, keys: {data.keys()}")
except Exception as e:
    print(f"PPUBS Error: {e}")

# 3. Test Lens.org public search
try:
    lens_url = "https://api.lens.org/patent/search"
    # without token
    req = urllib.request.Request(lens_url, data=b'{}', headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        print(f"Lens status: {resp.status}")
except Exception as e:
    print(f"Lens Error: {e}")
