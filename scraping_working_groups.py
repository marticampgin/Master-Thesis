import requests
import os
import pandas as pd
from bs4 import BeautifulSoup


def extract_wgs(url):
    # Parsing the URL
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Extracting all the tables
    tables = soup.find_all('table', class_='table')

    # List of all active working groups
    wgs = []

    # In each table extract all rows, in each row extact the first element, which as a WG
    for table in tables:
        rows = table.tbody.find_all("tr")
        for row in rows:
            columns = row.find_all("td")
            wg = columns[0].text.strip()
            wgs.append(wg)

    return wgs


def active_groups_in_files(filenames, wgs):
    # Extracting WGs that are present in our data
    active_groups_files = []
    for filename, extension in filenames:
        for wg in wgs:
            if filename == wg:
                active_groups_files.append(wg + "." + extension)
            
    return active_groups_files


# Might redo the function later
def combine_data(active_groups_files, additional_files, path_to_data, ratio=0.15):
    # Saving the name of the file in the a tuple for later use in analysis
    active_dfs = [(pd.read_csv(os.path.join(path_to_data, act_file)), act_file) for act_file in active_groups_files]
    active_bodies = [(df[["Body"]].sample(frac=ratio, replace=False), act_file) for df, act_file in active_dfs]

    additional_dfs = [(pd.read_csv(os.path.join(path_to_data, add_file)), add_file) for add_file in additional_files]
    additional_bodies = [(df[["Body"]].sample(frac=ratio, replace=False), add_file) for df, add_file in additional_dfs]

    # For now, simply create one big df will all the data
    df = active_bodies[0][0]
    for active_body, _ in active_bodies[1:]:
        df = pd.concat([df, active_body])

    for add_body, _ in additional_bodies:
        df = pd.concat([df, add_body])

    return df["Body"].tolist()

def main():
    import mailparser
    URL = r"https://datatracker.ietf.org/wg/#INT"
    path_to_email_arc = "email-archives/"

    wgs = extract_wgs(URL)
    all_csv_files = os.listdir(path_to_email_arc)
    filenames = [doc.rsplit(".", 1) for doc in all_csv_files]

    active_groups_files = active_groups_in_files(filenames, wgs)
    additional_files = ["ietf.csv", "architecture-discuss.csv"]  # manually adding additional wg-files

    email_bodies = combine_data(active_groups_files, additional_files, path_to_data=path_to_email_arc)
        
if __name__ == "__main__":
    main()

