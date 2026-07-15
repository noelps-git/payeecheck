"""
generate_dataset.py — PayeeCheck synthetic dataset v2.

400 transactions across 12 distinct scenario types covering every signal
module and demo use case. Kept small intentionally.

Run: python data/generate_dataset.py --seed 42
"""
import random, csv, json, argparse
from datetime import datetime, timedelta

def build(seed=42):
    random.seed(seed)
    FIRST = ["Suresh","Ramesh","Mohammed","Priya","Ananya","Vikram","Deepa",
             "Arjun","Rohan","Kavya","Sanjay","Lakshmi","Karthik","Meera",
             "Rajesh","Divya","Manoj","Pooja","Amit","Sneha"]
    LAST  = ["Kumar","Sharma","Patel","Reddy","Nair","Iyer","Singh","Gupta",
             "Rao","Menon","Das","Joshi","Verma","Pillai","Shah"]
    BIZ   = ["Enterprises","Trading","Solutions","Services","Associates"]
    REAL_MERCHANTS = [
        ("Amazon Pay","amazon.pay@apl"),("Paytm","paytm@paytm"),
        ("PhonePe","phonepe@ybl"),("Google Pay","googlepay@okaxis"),
        ("Flipkart","flipkart@icici"),("Swiggy","swiggy@ybl"),
        ("Zomato","zomato@paytm"),("BookMyShow","bookmyshow@hdfcbank"),
        ("IRCTC","irctc@sbi"),
    ]
    PSPS = ["ybl","okaxis","paytm","oksbi","icici","apl","hdfcbank"]
    FAKE_PSPS = ["axl","yba","0kaxis","paytrn","hdfcb"]
    SCAM_KW = ["support","help","verify","refund","kyc","service","secure"]
    BANKS = ["HDFC","SBI","ICICI","Axis","PNB","Kotak","BOB"]
    SANCTION_NAMES = ["Global Trade Finance Ltd","Eastern Star Holdings",
                      "Crescent Capital Group","Northern Light Enterprises",
                      "Pacific Rim Associates"]
    ABBR = [("State Bank of India","SBI"),("HDFC Bank","HDFC"),
            ("Punjab National Bank","PNB"),("Bank of Baroda","BOB")]

    def rn(): return f"{random.choice(FIRST)} {random.choice(LAST)}"
    def rv(h=None):
        h = h or random.choice(PSPS)
        return f"{random.choice(FIRST).lower()}.{random.randint(1,999)}@{h}"

    rows = []
    tx_id = 1

    def R(label,en,an,vpa,amt,input_method="type",vpa_age=500,ptx=100,
          us7=3,mf=False,mfc=0,lookalike=False,ns=0.95,ring="",
          bf=0,burst=False,session_anom=False,new_vpa=False,sanction=False):
        nonlocal tx_id
        r = {"tx_id":tx_id,"entered_name":en,"actual_name":an,
             "payee_vpa":vpa,"amount":round(amt,2),"input_method":input_method,
             "vpa_age_days":vpa_age,"prior_tx_count":ptx,
             "unique_senders_7d":us7,"mule_flagged":mf,"mule_flag_count":mfc,
             "is_lookalike":lookalike,"name_score":ns,"fraud_ring_id":ring,
             "bank_flags":bf,"seasonal_burst":burst,"session_anomaly":session_anom,
             "vpa_is_new":new_vpa,"sanction_match":sanction,"label":label}
        tx_id += 1
        return r

    # 1. CLEAN (80)
    for _ in range(80):
        if random.random()<0.4:
            name,vpa=random.choice(REAL_MERCHANTS)
        else:
            name,vpa=rn(),rv()
        rows.append(R("clean",name,name,vpa,random.uniform(100,50000),
            vpa_age=random.randint(300,2000),ptx=random.randint(50,1000),
            us7=random.randint(1,10),ns=round(random.uniform(0.88,1.0),2)))

    # 2. MULE single bank (30)
    for _ in range(30):
        name=rn()
        rows.append(R("mule",name,name,rv(),random.uniform(5000,150000),
            mf=True,mfc=1,us7=random.randint(15,60),
            vpa_age=random.randint(10,120),ptx=random.randint(5,40),bf=1))

    # 3. MULE consortium 2+ banks (20)
    for _ in range(20):
        fc=random.randint(2,4)
        name=rn()
        rows.append(R("mule_consortium",name,name,rv(),random.uniform(10000,200000),
            mf=True,mfc=fc,us7=random.randint(20,80),
            vpa_age=random.randint(5,90),ptx=random.randint(1,20),bf=fc))

    # 4. LOOKALIKE VPA (40)
    brands=["hdfc","sbi","icici","paytm","amazon","flipkart","phonepe"]
    for _ in range(40):
        brand=random.choice(brands)
        fvpa=f"{brand}.{random.choice(SCAM_KW)}@{random.choice(FAKE_PSPS)}"
        mf_=random.random()<0.4
        rows.append(R("lookalike",
            f"{brand.title()} {random.choice(SCAM_KW).title()}",rn(),fvpa,
            random.uniform(500,25000),input_method="paste",
            vpa_age=random.randint(1,15),ptx=random.randint(0,5),
            us7=random.randint(5,30),lookalike=True,
            ns=round(random.uniform(0.15,0.45),2),
            mf=mf_,mfc=1 if mf_ else 0))

    # 5. CLIPBOARD SCAM (25)
    for _ in range(25):
        bank=random.choice(BANKS)
        fvpa=f"{bank.lower()}.support@{random.choice(FAKE_PSPS)}"
        rows.append(R("clipboard_scam",f"{bank} Support",rn(),fvpa,
            random.uniform(5000,50000),input_method="paste",
            vpa_age=random.randint(1,20),ptx=random.randint(0,3),
            us7=random.randint(10,50),lookalike=True,
            ns=round(random.uniform(0.1,0.3),2),session_anom=True))

    # 6. BIZ vs INDIVIDUAL mismatch (30)
    for _ in range(30):
        ind=rn()
        biz=f"{random.choice(FIRST)} {random.choice(BIZ)}"
        rows.append(R("biz_individual_mismatch",biz,ind,rv(),
            random.uniform(5000,200000),
            vpa_age=random.randint(200,1500),ptx=random.randint(20,200),
            ns=round(random.uniform(0.45,0.72),2)))

    # 7. ABBREVIATION mismatch (20)
    for _ in range(20):
        full,abbr=random.choice(ABBR)
        rows.append(R("name_mismatch_abbr",abbr,full,rv(),
            random.uniform(1000,100000),
            vpa_age=random.randint(500,2000),ptx=random.randint(100,500),
            ns=round(random.uniform(0.22,0.45),2)))

    # 8. FRESH VPA high amount (20)
    for _ in range(20):
        name=rn()
        rows.append(R("fresh_vpa",name,name,
            f"{name.split()[0].lower()}.new@{random.choice(PSPS)}",
            random.uniform(50000,500000),
            vpa_age=random.randint(1,7),ptx=0,us7=1,new_vpa=True,
            ns=round(random.uniform(0.9,1.0),2)))

    # 9. FRAUD RINGS — 5 rings x 6 accounts (30)
    for ring_id in range(5):
        rname_=f"ring_{ring_id:02d}"
        rows.append(R("ring_member",rn(),rn(),rv(),
            random.uniform(20000,100000),
            mf=True,mfc=random.randint(1,3),us7=random.randint(20,60),
            vpa_age=random.randint(30,90),ring=rname_,
            bf=random.randint(1,3)))
        for _ in range(5):
            rows.append(R("ring_satellite",rn(),rn(),rv(),
                random.uniform(5000,50000),
                us7=random.randint(8,25),vpa_age=random.randint(15,60),
                ring=rname_,ns=round(random.uniform(0.85,1.0),2)))

    # 10. SANCTION MATCH (5)
    for sn in SANCTION_NAMES:
        rows.append(R("sanction",sn,sn,rv(),
            random.uniform(100000,2000000),
            vpa_age=random.randint(100,500),ptx=random.randint(5,50),
            sanction=True,ns=round(random.uniform(0.9,1.0),2)))

    # 11. SEASONAL BURST legitimate (15)
    for _ in range(15):
        name=rn()
        rows.append(R("seasonal_burst",name,name,rv(),
            random.uniform(10000,80000),
            us7=random.randint(40,100),vpa_age=random.randint(365,1500),
            ptx=random.randint(200,2000),burst=True,
            ns=round(random.uniform(0.9,1.0),2)))

    # 12. SESSION ANOMALY (10)
    for _ in range(10):
        name=rn()
        rows.append(R("session_anomaly",name,name,rv(),
            random.uniform(1000,30000),input_method="paste",
            vpa_age=random.randint(200,800),ptx=random.randint(30,200),
            session_anom=True,ns=round(random.uniform(0.88,1.0),2)))

    random.shuffle(rows)
    stats={r["label"]:stats.get(r["label"],0)+1 for r in rows
           for stats in [{}]}
    # recount
    stats={}
    for r in rows:
        stats[r["label"]]=stats.get(r["label"],0)+1
    return rows, stats

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    rows,stats=build(seed=args.seed)
    out="data/synthetic_transactions_v2.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("data/dataset_stats_v2.json","w") as f:
        json.dump({"total":len(rows),"by_label":stats,"seed":args.seed},f,indent=2)
    print(f"Generated {len(rows)} rows -> {out}")
    for k,v in sorted(stats.items()):
        print(f"  {k:30s} {v:4d}")
