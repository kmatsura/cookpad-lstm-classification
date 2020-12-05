import mysql.connector as mydb


if __name__ == "__main__":
    conn = mydb.connect(
        host='hostname',
        port='3306',
        user='kmatsuura',
        password='password',
        database='dbname'
    )
