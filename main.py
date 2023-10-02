import matplotlib.pyplot as plt
import seaborn as sns

from ietf_wg_mb_extractor import IETF_WG_MB_Extractor
from argparse import ArgumentParser
from explore_data import DataExplorer
from prepare_data import DataPreparator


def set_parameters(grid_color, axes_edgecolor, axes_facecolor):
    sns.set(rc={"grid.color": grid_color, 
                "axes.edgecolor": axes_edgecolor, 
                "axes.facecolor": axes_facecolor,
                'figure.figsize':(9, 5)})
    
def main():
    set_parameters(grid_color="#99e2b4",
                   axes_edgecolor="#99e2b4",
                   axes_facecolor="#e6ffed")
    
    parser = ArgumentParser()
    parser.add_argument('--archive_path', default='email-archives/')

    args = parser.parse_args()

    extractor = IETF_WG_MB_Extractor(archive_path=args.archive_path)
    extractor.combine_wg_files(ratio=None)
    active_wg_dataframe = extractor.get_combined_wg_dataframes()

    data_explorer = DataExplorer(colors = ['#eee82c', '#91cb3e', '#17A72D', '#4c934c','#368245'])
    data_preparator = DataPreparator(extractor)

    clean_wg_dataframe = data_preparator.emails_df_cleaning(active_wg_dataframe)

    # data_explorer.subject_count(clean_wg_dataframe, n_most_common=20)

    processed_bodies = data_preparator.preprocess_bodies()

    text_collection_dict = data_preparator.wg_combined_bodies_to_dict(processed_bodies)

    # data_explorer.most_commonly_used_words_per_wg(text_collection_dict, 'netmod', 15)

if __name__ == '__main__':
    main()