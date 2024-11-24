import os
import argparse
import numpy as np
import pandas as pd
from mysql import connector


"""
python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert384\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/all-MiniLM-L12-v2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert384\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/all-MiniLM-L12-v2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_cossim\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_cos-sim_50_512_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_cossim\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_cos-sim_50_512_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_simclr\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_sim-clr_50_512_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_simclr\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_sim-clr_50_512_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_nceclr\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_norm-temp-ce-sum_50_512_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper384_mean_nceclr\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-384_mean_norm-temp-ce-sum_50_512_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisbert384_mean_cossim\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisbert-384_mean_cos-sim_50_480_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisbert384_mean_cossim\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisbert-384_mean_cos-sim_50_480_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisbert384_mean_nceclr\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisbert-384_mean_norm-temp-ce-mean_50_480_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisbert384_mean_nceclr\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisbert-384_mean_norm-temp-ce-mean_50_480_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whispa384_cs\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whispa-384_mean_cos-sim_50_600_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whispa384_cs\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whispa-384_mean_cos-sim_50_600_1e-5_1e-2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whispa384_nce\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whispa-384_mean_nce-sum_50_900_1e-5_1e-2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whispa384_nce\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whispa-384_mean_nce-sum_50_900_1e-5_1e-2/hitop_embeddings.csv

python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert768\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/all-mpnet-base-v2/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$sbert768\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/all-mpnet-base-v2/hitop_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper768\$wtc_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-768/wtc_embeddings.csv
python sql/fill_table.py -c ~/.my.cnf -t "feat\$whisper768\$hitop_seg_persona\$user_id" --csv /cronus_data/rrao/WhiSBERT/embeddings/whisper-768/hitop_embeddings.csv
"""


def main():
    parser = argparse.ArgumentParser(description='DLATK Feature Table Filler')

    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the SQL Table to insert to")
    parser.add_argument("--csv", required=True, type=str, help="Specify the path to the features file (must be .csv format)")
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
        segments_path = '/cronus_data/rrao/hitop/seg_persona.csv'
        group_id_dtype = 'VARCHAR(24)'
    elif os.path.basename(csv_path).startswith('wtc'):
        segments_path = '/cronus_data/rrao/wtc_clinic/seg_persona.csv'
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

    df = pd.read_csv(segments_path)[['user_id', 'message_id']].merge(pd.read_csv(csv_path), on='message_id', how='left')

    if no_agg:
        for idx, row in df.iterrows():
            print(f'[{idx + 1}/{len(df)}]\tmessage_id: {row["message_id"]}')
            for feat_name, value in row[2:].items():
                values = (row['message_id'], feat_name, value, value)
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


if __name__ == '__main__':
    main()