import matplotlib.pyplot as plt
import random
import pandas as pd
import warnings
import re 
import statistics

from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Type, Tuple
from scipy.stats import norm

class DataExplorer:
    """
    As the name suggests, this class is responsible for providing
    different means of exploring and analyzing processed data. 
    It can produce things like WordClouds, distribution plots, 
    term-document matrices. For more detailed info, please check the 
    class methods down below.
    """

    def __init__(self, colors: List[str]) -> None:
        self.colors = colors


    def subject_count(self, dataframe: Type[pd.DataFrame], n_most_common: int) -> None:
        """
        A simple method that counts and displays the n most
        common subjects in the dataframe of concatenated emails. 
        """

        print("-----EMAIL ARCHIVE SUBJECT INFORMATION-----", end='\n\n')
        subject_cntr = Counter(dataframe["Subject"].tolist())
        print(f"Total subjects: {len(subject_cntr)}", end='\n\n')

        print(f'{n_most_common} most common subjects:')
        for subject, count in subject_cntr.most_common(n_most_common):
            print(f"Subject: {subject}  |  count: {count}")

        print()
    

    def wg_most_commonly_used_words(self, 
                                    token_collection: dict[str, List[str]], 
                                    wg: str, 
                                    n_tokens: int=20) -> None:
        """
        This method count the n most commonly used words for a given WG
        and produces a bar-plot visualization for that. 
        """

        tokens = token_collection[wg]

        most_common_tokens, counts = [], []
    
        cntr = Counter(tokens).most_common(n_tokens)  # Using Counter to count most common words

        for token, count in cntr:
            most_common_tokens.append(token)
            counts.append(count)

        plt.figure(figsize=(8, 4))
        plt.bar(most_common_tokens, 
                counts, 
                color=random.choices(self.colors, k=n_tokens))
        
        plt.title(f"Top {n_tokens} most commonly used words in {wg.upper()}")

    
    def messages_per_wg(self, 
                        text_collection: dict[str, List[str]], 
                        top_n: int=50,
                        histogram: bool=False,
                        bins: int=50) -> None:
        """
        This method simply count and plots total number of messages per 
        each working group
        """

        bodies_counts = [(wg, len(bodies)) for wg, bodies in text_collection.items()]
        bodies_counts.sort(key = lambda x: x[1], reverse=True)
        bodies_counts = bodies_counts[:top_n]

        if histogram:
            num_messages = [pair[1] for pair in bodies_counts]
            plt.figure(figsize=(9, 5))
            plt.hist(num_messages, 
                     bins=bins, 
                     density=True, 
                     alpha=0.8,
                     edgecolor='black', 
                     color='royalblue')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Probability Density')
        
        else:
            plt.figure(figsize=(9, 5))
            plt.bar([pair[0] for pair in bodies_counts],
                    [pair[1] for pair in bodies_counts],
                    color = "royalblue",
                    #align='edge',
                    edgecolor='black')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Number of messages')

        return bodies_counts


    def wg_wordcloud(self, 
                     text_collection: dict[str, List[str]], 
                     wg: str, 
                     max_words: int) -> None:
        """
        This method count the n most commonly used words for a given WG
        and produces a WordCloud-visualization for that. 
        """

        tokens = ' '.join(text_collection[wg])
        wordcloud = WordCloud(max_font_size=50,
                              background_color='white',
                              max_words=max_words).generate(tokens)

        plt.title(f'{max_words}-word WordCloud for {wg.upper()} WG')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

    
    def wg_body_len_dist(self, 
                         text_collection: dict[str, List[str]], 
                         wg: bool=None, 
                         histogram: bool=False, 
                         bins: int=45, 
                         whole: bool=False,
                         limit: List[int]=[0, 5000],
                         avg_per_group=False) -> None:
        """
        This method produces and plots a frequency distribution based on lenght of email bodies, given
        desired WG or whole collection of WGs. 
        """
        warnings.filterwarnings("ignore")

        if avg_per_group:
            avg_msg_lenghts_per_wg = []
            for wg, body_coll in text_collection.items():
                msg_lengths = [len(body.split()) for body in body_coll]
                mean = statistics.mean(msg_lengths)
                avg_msg_lenghts_per_wg.append(mean)
        
            plt.figure(figsize=(9, 5))
            plt.hist(avg_msg_lenghts_per_wg, 
                     bins=bins, 
                     density=True, 
                     alpha=0.8,
                     edgecolor='black', 
                     color='royalblue')
            plt.xlim(limit)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Probability Density')

            return sorted(avg_msg_lenghts_per_wg, reverse=True) 

        # In case we are interested in the whole collection (all WG bodies)
        if whole:
            body_lengths = [len(body.split()) 
                            for body_coll in text_collection.values()
                            for body in body_coll]
                        
        # In case we are interested in body lenghts of certain WG
        else:
            body_lengths = [len(body.split()) for body in text_collection[wg]]

        if histogram:
            plt.figure(figsize=(9, 5))
            plt.hist(body_lengths, 
                     bins=bins, 
                     density=True, 
                     alpha=0.8,
                     edgecolor='black', 
                     color='royalblue')
            plt.xlim(limit)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Probability Density')

        return sorted(body_lengths, reverse=True)

    
    def ngram_vectorizer(self, 
                         text_collection: dict[str, List[str]], 
                         vectorizer_type: str ='count', 
                         wgs: List[str]=[], 
                         top_n: int=20, 
                         ngram_range: Tuple[int, int]=(1,1), 
                         min_df: int=2, 
                         sent_lex: bool=False) -> None:
        """
        This method produces and plots a term-document matrix, based on type of vectorizer (either count or tf-idf), and whether we 
        are considering unigrams, bigrams or more genrally n-grams. 
        """

        # If looking for tokens from Sentiment Lexcion. This parameter should always be set to False, unless we 
        # are interested in the words from a Sentiment Lexicon 
        if sent_lex: 
            str_documents = [' '.join(word for word in doc) for doc in text_collection][:len(wgs)]

        else:
            documents = [bodies for desired_wg in wgs for wg, bodies in text_collection.items() if wg==desired_wg]
            str_documents = [' '.join(word for word in doc) for doc in documents]

        if vectorizer_type == 'tf_idf':
            vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(ngram_range=ngram_range)

        # Obtaining vectors
        vectors = vectorizer.fit_transform(str_documents)
        
        tdm = pd.DataFrame(vectors.todense().round(3))
        
        tdm.columns = vectorizer.get_feature_names_out()
        tdm = tdm.T
        tdm.columns = [wg for wg in wgs]

        if vectorizer_type == 'tf_idf':
            tdm['highest_score'] = tdm.max(axis=1)
            # For better display experience, sort everything by highest score
            tdm = tdm.sort_values(by='highest_score', ascending=False)

        elif vectorizer_type == 'count':
            tdm['total_count'] = tdm.sum(axis=1)
            # For better display experience, sort everything by total count
            tdm = tdm.sort_values(by='total_count', ascending=False)
        
        print(tdm.head(top_n))
        

    def sent_lex_vectorizer(self, 
                            text_collection: dict[str, List[str]], 
                            lexicon_path: str='vader_lexicon.txt',
                            wgs: List[str] = [], 
                            pos_thres: int=2, 
                            neg_thresh: int=-2) -> None:
        """
        This method extract words from a Sentiment Lexicon (this method is tailored for Vader Lexicon: 
        https://github.com/cjhutto/vaderSentiment) above or below a cerattain positivity/negativity threshold and
        then uses ngram_vectorizer() to crate a term-document matrix based on these extracted words. 
        
        """
        sentiment_lexicon = {}
        documents = [bodies for desired_wg in wgs for wg, bodies in text_collection.items() if wg==desired_wg]
        # Processing lexicon and extracting the words
        with open(lexicon_path, 'r') as lexicon:
            lexicon_lines = [re.split('\t', line.strip()) for line in lexicon]
    
        for token, score, _, _ in lexicon_lines: 
            if float(score) > pos_thres:
                sentiment_lexicon[token] = 'pos'
            
            elif float(score) < neg_thresh: 
                sentiment_lexicon[token] = 'neg'

        sent_lex_docs = []
        # Going through every email body and look for extracted sentiment words
        for doc in documents:
            transformed_doc = []
            for word in doc:
                if word in sentiment_lexicon.keys():
                    transformed_doc.append(word + '_' + sentiment_lexicon[word])
            sent_lex_docs.append(transformed_doc)
        
        # Creating and plotting term-document matrix
        self.ngram_vectorizer(text_collection=sent_lex_docs,
                              vectorizer_type='count',
                              wgs=wgs,
                              ngram_range=(1,1),
                              min_df=1,
                              sent_lex=True)
        

    def keyword_concordance(self, 
                            text_collection: dict[str, List[str]], 
                            wgs: List[str], 
                            keywords: List[str], 
                            left_context: int, 
                            right_context: int, 
                            max_num_samples: int) -> None:
        """
        This method takes keywords and outputs a concordance (context of given length around the keyword).
        This is done for each WG, and is limited by the amount of samples to extract for each keyword.
        
        """

        # To make the code more compact, extract relevant WGs here
        desired_wgs = [(wg, bodies) for desired_wg in wgs 
                                    for wg, bodies in text_collection.items() 
                                    if desired_wg == wg] 
        
        all_samples = []
        for keyword in keywords:
            keyword_samples = [] 
            for wg, bodies in desired_wgs:
                for body in bodies:
                    body = body.split()
                    for i, word in enumerate(body):
                        if keyword == word:
                            # Make sure the context is not out of list range
                            if i >= left_context and (i + right_context) < len(body):
                                # Extract context
                                left = body[i - left_context:i]
                                right = body[i+1:i+right_context + 1] 
                                whole = left + ['$'+word+'$'] + right
                    
                                # Combine context 
                                keyword_samples.append([wg, keyword, whole])

                                if len(keyword_samples) == max_num_samples:
                                    all_samples.extend(keyword_samples)
                                    break
                    # Breaking out of body loop
                    if len(keyword_samples) == max_num_samples:
                        break
                # Breaking out of WG loop
                if len(keyword_samples) == max_num_samples:
                    break
                                                
        
        print(f'Concordandce for the keywords {keywords} (+{left_context} keyword +{right_context})', end='\n\n')
        for wg, keyword, context in all_samples:
            print(f'WG: {wg.upper()}')
            print(' '.join(word for word in context))
            print("------------------------------", end='\n\n')
