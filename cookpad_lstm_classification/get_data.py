import os
import pandas as pd
import mysql.connector as mydb
from dotenv import load_dotenv


def get_title_and_category(conn):
    """
    cookpad_dataからレシピ名とそれに対応するカテゴリを持ってくる。
    """
    cur = conn.cursor()
    category_id = 'c910ca8994e8f6953e0b85bbfc73f8305274886d'
    sql = "SELECT r.title, sc.title FROM recipes r INNER JOIN search_category_recipes scr ON r.id = scr.recipe_id INNER JOIN search_categories sc ON scr.search_category_id = sc.id;"
    result = cur.execute(sql)
    rows = cur.fetchall()
    return result, rows


def main():
    load_dotenv(verbose=True)
    conn = mydb.connect(
        host=os.environ.get("HOST_NAME"),
        user=os.environ.get("USER_NAME"),
        password=os.environ.get("PASSWORD"),
        database=os.environ.get("DB_NAME")
    )
    conn.ping(reconnect=True)
    assert conn.is_connected(), "connection error"
    result, rows = get_title_and_category(conn)
    if result == 0:
        print("No Data")
    tmp_dict = {}
    for i, row in enumerate(rows):
        title, category_name = row
        tmp_dict[i] = title, category_name  # df.append()は遅いので、dictを使う。
    datasets = pd.DataFrame.from_dict(
        tmp_dict, orient='index', columns=["title", "category"])
    datasets = datasets.sample(frac=1).reset_index(drop=True)  # データフレームシャッフル
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTDIR = os.path.join(BASEDIR, 'data/reciepe_category/')
    FILENAME = "recipe_category_datasets.csv"
    if not os.path.exists(OUTPUTDIR):
        os.makedirs(OUTPUTDIR)
    datasets.to_csv(os.path.join(OUTPUTDIR, FILENAME))


if __name__ == "__main__":
    main()
