import sqlite3, os, json
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "biz.db")
c = sqlite3.connect(p); c.row_factory = sqlite3.Row
q = lambda s, *a: c.execute(s, a).fetchone()[0]
print("tickets", q("select count(*) from tickets"))
print("overdue", q("select count(*) from tickets where status!='已解决' and sla_deadline < datetime('now','localtime')"))
print("orders", q("select count(*) from orders"))
print("recent7", q("select count(*) from orders where created_at >= datetime('now','localtime','-7 day')"))
print("status", {r["status"]: r["n"] for r in c.execute("select status,count(*) n from orders group by status")})
print("low", q("select count(*) from inventory where qty<safety_stock"))
print("chunks", q("select count(*) from policies"))
print("sample ticket", dict(c.execute("select * from tickets limit 1").fetchone()))
print("sample order", dict(c.execute("select * from orders order by created_at desc limit 1").fetchone()))
