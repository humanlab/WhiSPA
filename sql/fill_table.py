import os, json
import argparse
import numpy as np
import pandas as pd
from mysql import connector


DEBUG = True
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


"""
python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert384\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/wtc_all-MiniLM-L12-v2.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert384\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/hitop_all-MiniLM-L12-v2.csv
"""


def main():
    parser = argparse.ArgumentParser(description='DLATK Feature Table Filler')

    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the SQL Table to insert to")
    parser.add_argument("--csv", required=True, type=str, help="Specify the path to the features (must be .csv file format)")
    parser.add_argument("--no_agg", action="store_true", help="Flag to disallow aggregation of features by `user_id`")
    args = parser.parse_args()

    # Create a connection to the MySQL server
    connection, cursor = get_sql_credentials(args.credential)
    driver(connection, cursor, args.table_name, args.csv, args.no_agg)
    

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
        database='HiTOP'
    )
    return connection, connection.cursor()


def driver(connection, cursor, table_name, csv_path, no_agg):
    if os.path.basename(csv_path).startswith('hitop'):
        segments_path = '/cronus_data/rrao/hitop/segment_outcomes.csv'
        group_id_dtype = 'VARCHAR(24)'
    elif os.path.basename(csv_path).startswith('wtc'):
        segments_path = '/cronus_data/rrao/wtc_clinic/segment_outcomes.csv'
        group_id_dtype = 'VARCHAR(24)' if no_agg else 'INT'
    try:
        create_query = f"""
        CREATE TABLE {table_name} (
            group_id {group_id_dtype},
            feat VARCHAR(4),
            value DOUBLE,
            group_norm DOUBLE,
            KEY (group_id),
            KEY (feat)
        );
        """
        cursor.execute(create_query)
        connection.commit()
    except Exception as e:
        print(e)
        raise Exception(f'DLATK Feature Table [{table_name}] already exists!')

    # SQL query to insert data into feature table format
    insert_query = f"""
        INSERT INTO {table_name}
        (group_id, feat, value, group_norm)
        VALUES (%s, %s, %s, %s)
    """

    df = pd.read_csv(segments_path)[['user_id', 'segment_id']].merge(pd.read_csv(csv_path), on='segment_id', how='left')

    if no_agg:
        for idx, row in df.iterrows():
            print(f'[{idx + 1}/{len(df)}]\tsegment_id: {row["segment_id"]}')
            for feat_name, value in row[2:].items():
                values = (row['segment_id'], feat_name, value, value)
                cursor.execute(insert_query, values)
    else:
        user_ids = np.unique(df['user_id'])
        for idx, user_id in enumerate(user_ids):
            print(f'[{idx + 1}/{len(user_ids)}]\tuser_id: {user_id}')
            mean_feats = df[df['user_id'] == user_id].iloc[:, 2:].mean()

            if group_id_dtype == 'INT':
                user_id = int(user_id)

            for feat_name, value in mean_feats.items():
                values = (user_id, feat_name, value, value)
                cursor.execute(insert_query, values)

    # Commit the changes to the database
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()


def log(msg):
    if DEBUG:
        print(msg)


if __name__ == '__main__':
    main()