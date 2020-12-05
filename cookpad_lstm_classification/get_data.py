import os
import mysql.connector as mydb
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv(verbose=True)
    conn = mydb.connect(
        host=os.environ.get("HOST_NAME"),
        user=os.environ.get("USER_NAME"),
        password=os.environ.get("PASSWORD"),
        database=os.environ.get("DB_NAME")
    )
    conn.ping(reconnect=True)
    print("connection:", conn.is_connected())
    cur = conn.cursor()
    category_id = 'c910ca8994e8f6953e0b85bbfc73f8305274886d'
    sql = "select id from search_categories;"
    result = cur.execute(sql)
    rows = cur.fetchall()
    if result == 0:
        print("No Data")
    for row in rows:
        print(row)