import os
import argparse
import numpy as np
import pandas as pd
from mysql import connector


"""
python sql/update_table.py -c ~/.my.cnf -t "wtc_user_persona" --csv /cronus_data/rrao/wtc_clinic/wtc_outcomes.csv
"""


def main():
    parser = argparse.ArgumentParser(description='DLATK Feature Table Filler')

    parser.add_argument("-c", "--credential", required=True, type=str, help="Specify path to credential file")
    parser.add_argument("-t", "--table_name", required=True, type=str, help="Specify the name of the SQL Table to insert to")
    parser.add_argument("--csv", required=True, type=str, help="Specify the path to the features file (must be .csv format)")
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
        database='HiTOP'
    )
    return connection, connection.cursor()


def driver(connection, cursor, table_name, csv_path):
    # SQL query to insert data into feature table format
    update_query = f"""
        UPDATE {table_name}
        SET pcl_score = %s, pcl_r = %s, pcl_a = %s, pcl_n = %s, pcl_h = %s
        WHERE user_id = %s
    """

    outcomes_df = pd.read_csv(csv_path)
    outcomes_df = outcomes_df[['video_id', 'PCL_SCORE', 'PCL_R', 'PCL_A', 'PCL_N', 'PCL_H']]
    outcomes_df.replace({np.nan: None}, inplace=True)

    for idx, row in outcomes_df.iterrows():
        print(f'[{idx}] Updating video_id: {row["video_id"]}')
        
        # Execute the update query with values from the current row
        cursor.execute(update_query, (
            row['PCL_SCORE'], 
            row['PCL_R'], 
            row['PCL_A'], 
            row['PCL_N'], 
            row['PCL_H'], 
            row['video_id']
        ))

    # Commit the changes to the database
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()


if __name__ == '__main__':
    main()