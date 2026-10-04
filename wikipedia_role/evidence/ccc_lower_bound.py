import json,time,subprocess
from wdq import q,UA
EDS=[("vi",["Q881"]),("id",["Q252"]),("th",["Q869"]),("ko",["Q884","Q423"]),("ms",["Q833"]),("tr",["Q43"]),("pl",["Q36"]),
("cs",["Q213"]),("hu",["Q28"]),("fa",["Q794"]),("ro",["Q218"]),("uk",["Q212"]),("el",["Q41"]),("sv",["Q34"]),("fi",["Q33"]),
("ja",["Q17"]),("ceb",["Q928"]),("war",["Q928"]),("he",["Q801"]),("hi",["Q668"]),("bn",["Q902"])]
COUNTRY="wdt:P17|wdt:P27|wdt:P495|wdt:P1532"
def light(w,c):
    vals=" ".join("wd:"+x for x in c)
    return f"""SELECT (COUNT(DISTINCT ?item) AS ?n) WHERE {{
  ?s schema:about ?item ; schema:isPartOf <https://{w}.wikipedia.org/> .
  VALUES ?c {{ {vals} }}
  {{ ?item {COUNTRY} ?c }} UNION {{ ?item wdt:P131+ ?c }}
  FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q4167836 }} FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q11266439 }}
}}"""
def arts(w):
    for i in range(6):
        s=subprocess.run(["curl","-sS","-m","40","-A",UA,f"https://{w}.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"],capture_output=True,text=True).stdout
        try: return json.loads(s)['query']['statistics']['articles']
        except Exception: time.sleep(5*(i+1))
out={}
for w,c in EDS:
    a=arts(w); time.sleep(2)
    r=None
    for attempt in range(3):
        res,t=q(light(w,c),timeout=100)
        if not isinstance(res,tuple):
            r=int(res['results']['bindings'][0]['n']['value']); break
        time.sleep(15)
    out[w]={"qids":c,"articles":a,"ccc_wikidata_country_and_admin":r,"share":(round(r/a,4) if r and a else None),"seconds":round(t,1)}
    print(w,out[w],flush=True)
    json.dump(out,open("ccc_light_results.json","w"),indent=1)
    time.sleep(4)
