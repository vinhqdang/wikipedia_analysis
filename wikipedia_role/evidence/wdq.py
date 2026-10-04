import json,subprocess,time,urllib.parse,sys
UA="wikipedia-analysis-research (dqvinh87@gmail.com)"
def q(sparql,timeout=120):
    url="https://query.wikidata.org/sparql?format=json&query="+urllib.parse.quote(sparql)
    t=time.time()
    r=subprocess.run(["curl","-sS","-m",str(timeout),"-A",UA,"-H","Accept: application/sparql-results+json",url],capture_output=True,text=True)
    try: return json.loads(r.stdout),time.time()-t
    except Exception: return (r.stdout[:300],r.stderr[:200]),time.time()-t
if __name__=="__main__":
    print(q("SELECT (COUNT(*) AS ?n) WHERE { ?item wdt:P17 wd:Q881 . }"))
