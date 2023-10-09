import time
import re

from nltk.corpus import stopwords
from collections import defaultdict
# nltk.download('stopwords')


class DataPreparator:
    """
    A class that prepares/cleans dat a in various ways.
    Usually used for pre- (and potentially post-) processing
    of textual data, thus preparing it for further analysis. 
    """

    def __init__(self, extractor):
        self.dataframe = None
        self.extractor = extractor


    def emails_df_cleaning(self, dataframe):
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
        print(f'Dataframe shape after cleaning: {dataframe.shape}', end='\n\n')
        return dataframe
    

    def preprocess_bodies(self):

        """
        This method is resposible for peforming perhaps
        the most of the email processing, as it cleans the 
        email bodies. It utilizies a method of a separate
        IETF_WG_MB_Extractor-class. After utilizing extractor
        for pre-processing bodies, it then applies some
        post-processing, such as removing empty bodies and
        reformatting whitespaces, as well as providing
        some statistics after processing bodies of text.   
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
    

    def wg_bodies_to_dict(self, bodies):
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
    

    # There are several points of notice here: 

    # The number of messages varies drastically between each WG
    # It is probably wise to inlcude some set percent of data from 
    # Each working group, or somehow sample more samples from groups 
    # With very few messages, to make sure that they are included

    # Other problem concerns duplicates - there is a number of messages
    # that appear to be identical. Removing them would be good to prevent 
    # duplicates being both in training and validation data. 
    # However, it is unclear for now what to do with duplicates that 
    # Appear in different working groups: if we notice a duplicate in 
    # appearing in different working groups, which group shoud we delete it
    # from then? The choice will impact the sentiment analysis per wg afterwards 

    def prepare_data_for_model(self, text_collection, max_context_win_size=512):
        """
        This method further processes and prepairs data for being sent to
        a tokenizer/model. More specifically, it truncates texts longer than
        max. context window size and removes duplicate/noisy samples. 
        """
    
        for wg, bodies in text_collection.items():
            all_bodies = []
            for i, body in enumerate(bodies):
                body = body.split()
                if len(body) == 5:
                    print(i, wg, body)
                # Truncate bodies longer that 512 tokens
                # if len(body) > max_context_win_size:
                #    updated_text_collection[wg].append(" ".join(word for word in body[:max_context_win_size])) 

                # Removing noisy messages among those that are less that 4 words long
