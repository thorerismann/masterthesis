
import sqlite3

import pandas as pd


def test_database_structure(path):
    # Test if the database structure is as expected
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    print(tables)
    c.close()

path='/home/tge/masterthesis/app/temp5/geodata.db'
test_database_structure(path)

print('hi')

def test_table_contents(path, table):
    # Test if the table contents are as expected
    conn = sqlite3.connect(path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

datatables = [('fitnahtemp',), ('fitnahuhistreet',), ('dem',)]

fitnah = test_table_contents(path, 'fitnahtemp')
fitnahuhistreet = test_table_contents(path, 'fitnahuhistreet')
dem = test_table_contents(path, 'dem')
print('hello to debug')
print('last ol debug')