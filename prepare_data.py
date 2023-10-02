import time
import re

from nltk.corpus import stopwords
from collections import defaultdict
# nltk.download('stopwords')


class DataPreparator:
    def __init__(self, extractor):
        self.dataframe = None
        self.extractor = extractor


    def emails_df_cleaning(self, dataframe):
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
        print(f'Dataframe shape after cleaning: {dataframe.shape}', end='\n\n')
        return dataframe
    

    def preprocess_bodies(self):
        # ---------- PRE-PROCESSING ----------
        print("------PROCESSING------", end='\n\n')
        bodies = self.dataframe['Body'].tolist()
        wgs = self.dataframe['Working Group'].tolist()
        bodies_wgs = list(zip(bodies, wgs))

        start = time.time()

        processed_bodies_wgs, stats = self.extractor.process_email_bodies(bodies_wgs, 
                                                                          lower=True, 
                                                                          punc=True, 
                                                                          digits=True, 
                                                                          newline=True)     

        print(f"Prepocessing time: {time.time() - start:.2f} s.", end='\n\n')

        total_removed_bodies = sum(stats.values())
        print(f"Number of mails removed: {total_removed_bodies}")
        print("---------------------------------------")
        print(f"Encrypted messages: {stats['encryp']}")
        print(f"Ill-formated messages: {stats['ill_format']}")
        print(f"Announc. messages: {stats['announ']}")
        print(f"Unknown endcoding messages: {stats['unknown_enc']}")
        

        # ---------- POST-PROCESSING ---------- 
        # Removing empty messages, if any are present 
        empty_messages = 0
        processed_bodies_wgs_no_empty = []
        
        for body, wg in  processed_bodies_wgs:
            if body == '':
                empty_messages += 1
                continue
            processed_bodies_wgs_no_empty.append([body, wg])

        print(f"Empty messages (after pre-processing): {empty_messages}")
        print("---------------------------------------")
        print(f"Total number of mails after processing: {len(processed_bodies_wgs_no_empty)}")

        # Replacing 2+ whitespaces with 1 whitespace, and replacing underscore char. (_) with an empty string
        for i, body_wg in enumerate(processed_bodies_wgs_no_empty): 
            processed_bodies_wgs_no_empty[i][0] = re.sub('\s{2,}', ' ', body_wg[0])
            processed_bodies_wgs_no_empty[i][0] = re.sub('_+', '', body_wg[0])

        return processed_bodies_wgs_no_empty
    

    def wg_combined_bodies_to_dict(self, bodies):
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
    

    def wg_bodies_to_dict(self, bodies):
        text_collection = defaultdict(list)
        for body, wg in bodies:
            text_collection[wg].append(body)
        
        return text_collection