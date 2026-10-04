import json,time,subprocess,sys
UA="wikipedia-analysis-research (dqvinh87@gmail.com)"
def get(url):
    for i in range(6):
        r=subprocess.run(["curl","-sS","-m","60","-A",UA,url],capture_output=True,text=True).stdout
        try: return json.loads(r)
        except Exception: time.sleep(4*(i+1))
    return None
out={}
for w in ["vi.wikipedia.org","en.wikipedia.org"]:
    o={}
    for name,path in [("editors_5_99","editors/aggregate/%s/all-editor-types/content/5..99-edits/monthly/20250801/20260801"),
                      ("editors_100","editors/aggregate/%s/all-editor-types/content/100..-edits/monthly/20250801/20260801"),
                      ("newpages_all","edited-pages/new/%s/all-editor-types/content/monthly/20250801/20260801"),
                      ("newpages_user","edited-pages/new/%s/user/content/monthly/20250801/20260801"),
                      ("newpages_bot","edited-pages/new/%s/group-bot/content/monthly/20250801/20260801")]:
        d=get("https://wikimedia.org/api/rest_v1/metrics/"+path%w); time.sleep(2)
        try: o[name]=[(r['timestamp'][:7],r.get('editors',r.get('new_pages'))) for r in d['items'][0]['results']]
        except Exception: o[name]=None
    s=get(f"https://{w}/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"); time.sleep(2)
    o["stats"]=s['query']['statistics'] if s else None
    out[w]=o
json.dump(out,open("vi_en_pull.json","w"),indent=1)
for w,o in out.items():
    print(w,o["stats"])
    for k in o:
        if k!="stats" and o[k]: print(" ",k,o[k][-4:])
