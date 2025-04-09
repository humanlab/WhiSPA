import os
import argparse
import numpy as np
import pandas as pd
from mysql import connector
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='DLATK Feature Table Filler')
    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-d", "--database", required=True, type=str, help="Specify name of database")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the feature table to create")
    parser.add_argument("--csv", required=True, type=str, help="Specify the path to the features file (must be .csv format)")
    parser.add_argument("--no_agg", action="store_true", help="Flag to disallow aggregation of features by `user_id`")
    args = parser.parse_args()

    # Create a connection to the MySQL server
    connection, cursor = get_sql_credentials(args.credential, args.database)
    driver(connection, cursor, args.table_name, args.csv, args.no_agg)
    

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
        database=database
    )
    return connection, connection.cursor()


def driver(connection, cursor, table_name, csv_path, no_agg):
    group_id_dtype = 'VARCHAR(24)'
    if not no_agg:
        if os.path.basename(csv_path).startswith('hitop'):
            segments_path = os.getenv('HITOP_DATA_DIR')
        elif os.path.basename(csv_path).startswith('wtc'):
            segments_path = os.getenv('WTC_DATA_DIR')
            group_id_dtype = 'INT'
    
    try:
        create_query = f"""
        CREATE TABLE {table_name} (
            group_id {group_id_dtype},
            feat VARCHAR(5),
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


    if no_agg:
        segments_300 = pd.read_csv(f'{os.getenv("WHISPA_DIR")}whispa_affect_segments.csv')['message_id']

        wtc_emb_df = pd.read_csv(os.path.join(os.path.dirname(csv_path), 'wtc_embeddings.csv'))
        wtc_emb_df = wtc_emb_df[wtc_emb_df['message_id'].isin(segments_300)]
        hitop_emb_df = pd.read_csv(os.path.join(os.path.dirname(csv_path), 'hitop_embeddings.csv'))
        hitop_emb_df = hitop_emb_df[hitop_emb_df['message_id'].isin(segments_300)]

        df = pd.concat([hitop_emb_df, wtc_emb_df]).reset_index(drop=True)
        for idx, row in df.iterrows():
            print(f'[{idx + 1}/{len(df)}]  message_id: {row["message_id"]}')
            for feat_name, value in row[1:].items():
                values = (row['message_id'], feat_name, value, value)
                cursor.execute(insert_query, values)
    else:
        df = pd.read_csv(f'{segments_path}/whispa_dataset.csv')[['user_id', 'message_id']].merge(pd.read_csv(csv_path), on='message_id', how='left')
        user_ids = np.unique(df['user_id'])
        for idx, user_id in enumerate(user_ids):
            print(f'[{idx + 1}/{len(user_ids)}]  user_id: {user_id}')
            mean_feats = df[df['user_id'] == user_id].iloc[:, 2:].mean()

            if group_id_dtype == 'INT':
                user_id = int(user_id)

            for feat_name, value in mean_feats.items():
                values = (user_id, feat_name, value, value)
                cursor.execute(insert_query, values)

    # Commit the changes to the database
    connection.commit()

    print(f'\nCreated table `{table_name}` and finished inserting features...\n')

    # Close the cursor and connection
    cursor.close()
    connection.close()


if __name__ == '__main__':
    main()
