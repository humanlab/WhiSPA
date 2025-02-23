import os, re
import argparse
import numpy as np
import pandas as pd
from mysql import connector


DEBUG = True
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Script to Convert CSV to DLATK Feature Table (SQL)')
    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-d", "--database", required=True, type=str, help="Specify the name of the database")
    parser.add_argument("-t", "--table", required=True, type=str, help="Specify the name of the SQL Table")
    parser.add_argument("-g", "--group", required=False, type=str, help="Specify the group column name")
    parser.add_argument("--csv", required=True, type=str, help="Specify the .csv file")
    args = parser.parse_args()

    # Create a connection to the MySQL server
    connection, cursor = get_sql_credentials(args.credential, args.database)
    driver(connection, cursor, args.table, args.group, args.csv)
    

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

    connection = connector.connect(
        host='localhost',
        user=usr,
        password=pwd,
        database=database,
    )
    return connection, connection.cursor()
    

def driver(connection, cursor, table, group, csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8').dropna().reset_index(drop=True)

    group_id_dtype = 'VARCHAR(24)'
    if os.path.basename(csv_path).startswith('wtc'):
        group_id_dtype = 'INT'

    create_query = f"""
    CREATE TABLE {table} (
        group_id {group_id_dtype},
        feat VARCHAR({max([len(col) for col in df.columns if col != group])}),
        value DOUBLE,
        group_norm DOUBLE,
        KEY (group_id),
        KEY (feat)
    );
    """
    cursor.execute(create_query)

    insert_query = f"""
        INSERT INTO {table}
        (group_id, feat, value, group_norm)
        VALUES (%s, %s, %s, %s)
    """
    for idx, row in df.iterrows():
        print(f'[{idx + 1}/{len(df)}] {group} {row[group]}')
        for feat_name, value in row.items():
            if feat_name == group:
                continue
            values = (row[group], feat_name, value, value)
            cursor.execute(insert_query, values)
        # Commit the changes to the database
        connection.commit()

    print(f'\nCreated and populated feature table `{table}`...\n')

    # Close the cursor and connection
    cursor.close()
    connection.close()


if __name__ == '__main__':
    main()