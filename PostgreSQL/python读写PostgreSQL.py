
import psycopg2

conn = psycopg2.connect(database="trade", user="tradeopr", password="Opr6666!", host="localhost", port="5432")

cur = conn.cursor()
cur.execute('''select production_date from public.data_price_quantified_measure where uuid='edff9a83-4bc9-42da-8ae2-9537c18b8fa1' and flag = true;''')
rows = cur.fetchall()
for row in rows:
    print(row)

conn.commit()
conn.close()

