import os, re
import argparse
import numpy as np
import pandas as pd
from mysql import connector


DEBUG = True
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


"""
python sql/csv2sql.py -c ~/.my.cnf -t "wtc_aad" --csv code/aad_wtc.csv
python sql/csv2sql.py -c ~/.my.cnf -t "hitop_aad" --csv code/aad_hitop.csv
python sql/csv2sql.py -c ~/.my.cnf -t "wtc_uaad" --csv code/uaad_wtc.csv
python sql/csv2sql.py -c ~/.my.cnf -t "hitop_uaad" --csv code/uaad_hitop.csv

python sql/csv2sql.py -c ~/.my.cnf -t "wtc_seg_persona" --csv /cronus_data/rrao/wtc_clinic/whispa_dataset.csv \
&& python sql/csv2sql.py -c ~/.my.cnf -t "hitop_seg_persona" --csv /cronus_data/rrao/hitop/whispa_dataset.csv \
&& python sql/csv2sql.py -c ~/.my.cnf -t "wtc_user_persona" --csv /cronus_data/rrao/wtc_clinic/user_outcomes.csv \
&& python sql/csv2sql.py -c ~/.my.cnf -t "hitop_user_persona" --csv /cronus_data/rrao/hitop/user_outcomes.csv
"""


def main():
    parser = argparse.ArgumentParser(description='Script to Convert CSV to SQL file')

    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the SQL Table")
    parser.add_argument("--csv", required=True, type=str, help="Specify the .csv file")
    args = parser.parse_args()

    # Create a connection to the MySQL server
    connection, cursor = get_sql_credentials(args.credential)
    driver(connection, cursor, args.table_name, args.csv)
    

# Verifies user's credentials and returns a MySQL Connection object
def get_sql_credentials(filename):
    usr = ''
    pwd = ''
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith('user'):
                usr = line[5:].strip()
            if line.startswith('password'):
                pwd = line[9:].strip()

    connection = connector.connect(
        host='localhost',
        user=usr,
        password=pwd,
        database='HiTOP',
    )
    return connection, connection.cursor()


# Function to map pandas data types to MySQL data types
def map_dtype_to_mysql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'DATETIME'
    else:
        return 'TEXT'
    

def driver(connection, cursor, table_name, csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8').dropna().reset_index(drop=True)
    # df['message'] = df['message'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x) if isinstance(x, str) else x)

    # Dynamically create SQL columns based on DataFrame dtypes
    columns = []
    for col_name, dtype in df.dtypes.items():
        if col_name == 'user_id':
            if pd.api.types.is_integer_dtype(dtype):
                mysql_type = 'INT'
            else:
                mysql_type = 'VARCHAR(24)'
        elif col_name == 'message_id':
            mysql_type = 'VARCHAR(24)'
        elif col_name == 'filename':
            mysql_type = 'VARCHAR(100)'
        else:
            mysql_type = map_dtype_to_mysql(dtype)
        columns.append(f"{col_name} {mysql_type}")

    create_query = f"CREATE TABLE {table_name} ({', '.join(columns)});"
    cursor.execute(create_query)

    for idx, row in df.iterrows():
        print(f'[{idx + 1}/{len(df)}] Inserted: {row.values[0]}')
        placeholders = ', '.join(['%s'] * len(row))
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        cursor.execute(insert_query, tuple(row))

    # Close the cursor and connection
    cursor.close()
    connection.close()


if __name__ == '__main__':
    main()