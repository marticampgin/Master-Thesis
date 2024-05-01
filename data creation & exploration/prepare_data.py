import time
import re
import pandas as pd
import random 

from nltk.corpus import stopwords
from collections import defaultdict
from typing import Type, List
from ietf_wg_mb_extractor import IETF_WG_MB_Extractor
# nltk.download('stopwords')


class DataPreparator:
    """
    A class that prepares/cleans dat a in various ways.
    Usually used for pre- (and potentially post-) processing
    of textual data, thus preparing it for further analysis. 
    """

    def __init__(self, extractor: Type[pd.DataFrame]) -> None:
        self.dataframe = None
        self.extractor = extractor


    def emails_df_cleaning(self, dataframe: Type[pd.DataFrame]) -> pd.DataFrame:
        """
        This method drops empty rows and duplicates from
        a pandas dataframe of concatenated email messages.
        It then filters out messages older that 2 years, 
        before returning the dataframe. 
        """

        print(f'Initial dataframe shape: {dataframe.shape}')

        dataframe = dataframe.dropna(subset=["Body"])  # removing rows w. empty bodies
        dataframe = dataframe.dropna(subset=["Date"])  # removing rows w. empty dates
        dataframe = dataframe.drop_duplicates(subset=["Body"])  # removing rows w. duplicate bodies

        # Filtering out mails that are > 2 years old
        dataframe = dataframe.reset_index()  # adding int index
        dates = dataframe["Date"].tolist()

        indices_to_drop = []
        for i, date in enumerate(dates):
            if int(date[:4]) not in range(2021, 2023):  # if first 4 chars. (year) not in given range, drop the message
                indices_to_drop.append(i)

        dataframe = dataframe.drop(indices_to_drop)
        
        self.dataframe = dataframe
        print("-----------------------------------------")
        print(f'Dataframe shape after cleaning rows: {dataframe.shape}', end='\n\n')
        return dataframe
    

    def preprocess_bodies(self) -> List[List[str]]:

        """
        This method is resposible for peforming perhaps
        the most of the email processing, as it cleans the 
        email bodies. It utilizies a method of a separate
        IETF_WG_MB_Extractor-class. After utilizing extractor
        for pre-processing bodies, it prints some of the
        statistics after processing the texts.   
        """

        # ---------- PRE-PROCESSING ----------
        print("------PROCESSING------", end='\n\n')
        bodies = self.dataframe['Body'].tolist()

        wgs = self.dataframe['Working Group'].tolist()
        bodies_wgs = list(zip(bodies, wgs))

        start = time.time()

        # Utilizing processing method of IETF_WG_MB_Extractor-class
        processed_bodies_wgs, stats = self.extractor.process_email_bodies(bodies_wgs, 
                                                                          lower=True, 
                                                                          punc=False, 
                                                                          digits=False, 
                                                                          newline=True)     

        print(f"Prepocessing time: {time.time() - start:.2f} s.", end='\n\n')

        total_removed_bodies = sum(stats.values())
        print(f"Number of mails removed: {total_removed_bodies}")
        print("---------------------------------------")
        print(f"Encrypted messages: {stats['encryp']}")
        print(f"Ill from-formated messages: {stats['ill_from_format']}")
        print(f"Announc. messages: {stats['announ']}")
        print(f"Unknown endcoding messages: {stats['unknown_enc']}")
        print(f"Empty messages post-processing: {stats['empty_post_proc']}")
        print(f'Diff. language: {stats["diff_lang"]}')
        print(f'Diverse other noise: {stats["diverse_noise"]}')

        return processed_bodies_wgs
    

    def wg_combined_bodies_to_dict(self, bodies: List[List[str]]) -> dict[str, List[str]]:
        """
        For each WG, this method uses the WGs name as key and maps
        it to a combined collections of all tokens from all email bodies,
        belonging to that particular WG (excluding stopwords). This is 
        used for further analyis. 
        """

        stop_words = set(stopwords.words('english'))
        text_collection = {}

        # Populate dict with litst (could probably be optimized w. defaultdict or smth)
        wgs_set = set(wg for _, wg in bodies)
        for wg in wgs_set:
            text_collection[wg] = []

        for body, wg in bodies:
            body = body.split()
            for token in body:
                if token not in stop_words:
                    text_collection[wg].append(token)

        return text_collection
    

    def wg_bodies_to_dict(self, bodies: List[List[str]]) -> dict[str, List[str]]:
        """
        For each WG, this method uses the WGs name as key and maps
        it to a collection of all email bodies belonging to that particular WG
        without combining them and keeping each body separate. This is used for futher 
        analysis. 
        """

        text_collection = defaultdict(list)
        for body, wg in bodies:
            text_collection[wg].append(body)
        
        return text_collection
    

    def prepare_data_for_model(self, text_collection: dict[str, List[str]], 
                               max_context_win_size: int=512,
                               max_body_len : int=800, 
                               percent_of_data: float=0.14,
                               seed: int=7) -> dict[str, List[str]]:
        """
        This method further processes and prepairs data for being sent to
        a tokenizer/model. More specifically, for sentences consisting
        of < 6 tokens, it extracts only specific sentences, determined by
        the to_keep.txt file, containing manually extracted sentences to keep. 
        It also truncates messages that exceed 512 tokens. 
        """

        # Figure out a way to deal with duplicates in a robust manner.
        # For now, simply remove duplicates after extracting a percentage of 
        # data from each group and combining samples. 
        random.seed(seed)

        with open("short_messages_to_keep.txt", 'r') as f:
            to_keep = [line.rstrip().split(maxsplit=2) for line in f]

        processed_bodies = defaultdict(list)

        for wg, bodies in text_collection.items():
            for i, body in enumerate(bodies):
                body = body.split()
                
                if len(body) > max_body_len:
                    continue

                elif len(body) < 6:  # Only consider bodies of < 6 tokens long
                    for idx_wg_body in to_keep:
                        desired_id, desired_wg = idx_wg_body[0], idx_wg_body[1]
                        if int(desired_id) == i and desired_wg == wg:  # Only keeping manually extracted email bodies, rest is noise
                              processed_bodies[wg].append(" ".join(word for word in body))

                elif len(body) > max_context_win_size:
                    body = body[:max_context_win_size]  # Truncating bodies longer than 512 tokens
                    processed_bodies[wg].append(" ".join(word for word in body))

                # Simply append the body
                else:
                    processed_bodies[wg].append(" ".join(word for word in body))
                
    
        # Train data: for now, simply extract the same given percentage of data from each WG, then remove duplicates
        train_data = []
        for bodies_list in processed_bodies.values():
            n_samples = int(len(bodies_list) * percent_of_data)
            sampled_bodies = random.sample(bodies_list, k=n_samples)
            train_data.extend(sampled_bodies)

        print(f'Requested num. of samples: {len(train_data)}')
        train_data = list(set(train_data))
        print(f'Actual num. of samples (due to post duplicate-removal): {len(train_data)}')

        return train_data
        

        


                    
                 
