import os, json
import argparse
import numpy as np
import pandas as pd
from mysql import connector
from sqlalchemy import create_engine


DEBUG = True
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


"""
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "msgs_whisper_large_dia2_p1" --csv msgs_whisper_large_dia2_p1.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "msgs_whisper_large_dia2_p2" --csv msgs_whisper_large_dia2_p2.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "msgs_whisper_large_dia2_p3" --csv msgs_whisper_large_dia2_p3.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "feat\$cat_dd_depAnxAng_w\$wtc_seg_persona\$message_id\$1gra" --csv aad_wtc.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "feat\$cat_dd_depAnxAng_w\$hitop_seg_persona\$message_id\$1gra" --csv aad_hitop.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "feat\$cat_dd_affInt_w\$wtc_seg_persona\$message_id\$1gra" --csv va_wtc.csv
python sql/sql2csv.py -c ~/.my.cnf -d HiTOP -t "feat\$cat_dd_affInt_w\$hitop_seg_persona\$message_id\$1gra" --csv va_hitop.csv
"""


def main():
    parser = argparse.ArgumentParser(description='Script to Convert SQL to CSV file')
    parser.add_argument("-d", "--database", required=True, type=str, help="Specify the name of the database")
    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the SQL Table")
    parser.add_argument("--csv", required=True, type=str, help="Specify to the .csv file")
    args = parser.parse_args()

    # Create a connection to the MySQL server
    engine = get_sql_credentials(args.credential, args.database)
    driver(engine, args.table_name, args.csv)
    

# Verifies user's credentials and returns a MySQL Connection object
def get_sql_credentials(filename, database):
    usr = ''
    pwd = ''
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith('user'):
                usr = line[5:].strip()
            if line.startswith('password'):
                pwd = line[9:].strip()

    # connection = connector.connect(
    #     host='localhost',
    #     user=usr,
    #     password=pwd,
    #     database='HiTOP'
    # )
    # return connection, connection.cursor()
    engine = create_engine(f"mysql+mysqlconnector://{usr}:{pwd}@localhost/{database}")
    return engine


def driver(engine, table_name, csv_path):
    # SQL query to insert data into table
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    df.to_csv(csv_path, index=False)
    print(f'Done. File created: {csv_path}')

    # No need to explicitly close SQLAlchemy engine (handled automatically)
    # # Close the cursor and connection
    # cursor.close()
    # connection.close()


def log(msg):
    if DEBUG:
        print(msg)


if __name__ == '__main__':
    main()
