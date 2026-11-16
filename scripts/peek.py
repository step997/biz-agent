import sqlite3
p = r"D:\Projects\biz-agent\data\biz.db"
c = sqlite3.connect(p); c.row_factory = sqlite3.Row
print("customers:", [dict(r) for r in c.execute("select id,name,level from customers limit 8")])
print("hero orders:")
for r in c.execute("""select o.order_no,c.name,o.status,o.amount,o.created_at from orders o
                      join customers c on c.id=o.customer_id where o.customer_id<=3
                      order by o.created_at desc limit 12"""):
    print(" ", tuple(r))
print("names sample:", [r[0] for r in c.execute("select name from customers limit 12")])
