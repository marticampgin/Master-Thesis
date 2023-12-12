import requests
import os
import pandas as pd
import quopri
import re
import contractions

from tqdm import tqdm
from mailparser_reply import EmailReplyParser
from bs4 import BeautifulSoup
from typing import List, Type, Tuple


class IETF_WG_MB_Extractor:
    """
    IETF WORKING GROUP MAIL BODY EXTRACTOR
    --------------------------------------
    
    This class scrapes the names of currently active IETF
    working groups (WGs) https://datatracker.ietf.org/wg/,
    compares the names against locally stored
    mail-archive file names downloaded from 
    https://www.ietf.org/mail-archive/text/?C=M;O=D and stored
    as .csv files. 

    In the case names match, it implies that mail-archive
    data for that particular active WG is present locally.

    After assesing whether data for a particular active WG
    is avaliable, the class proceeds and pre-processes the
    mail bodies, making data ready to be fed to a downstream
    model, or used for a downstream task. 
    
    """
    def __init__(self, verbose: bool=True, archive_path: str="email-archives/") -> None:
        self.wgs_url = r"https://datatracker.ietf.org/wg/#INT"
        self.archive_path = archive_path
        self.active_wg_dataframes = None
        self.verbose = verbose
        self.lng = ["en", "de", "fr"]

    def _extract_wgs(self) -> List[str]:
        """
        A private method that scrapes all the active Working
        Groups (WGs) from the IETF-datatracker website by
        using Beautiful Soup.
        """
        
        # Parse the URL
        page = requests.get(self.wgs_url)
        soup = BeautifulSoup(page.content, "html.parser")
    
        # Extracting all the tables
        tables = soup.find_all('table', class_='table')
    
        # List of all active WGs
        wgs = []
    
        # In each table extract all rows, in each row extract the first element, which as a WG
        for table in tables:
            rows = table.tbody.find_all("tr")
            for row in rows:
                columns = row.find_all("td")
                wg = columns[0].text.strip()
                wgs.append(wg)

        if self.verbose:
            print("Successfully scraped active WG names",
                  end="\n------------------------------------\n")
            
        return wgs

    
    def _active_groups_in_files(self, filenames: List[List[str]], wgs: List[str]) -> List[str]:
        """
        A private method that compares all the scraped
        WG names against downloaded mail archives
        that include those names as filenames. 
        In case of a match, the name is added to a list.
        """
        
        active_wg_files = []
        for filename, extension in filenames:
            for wg in wgs:
                if filename == wg:
                    active_wg_files.append(wg + "." + extension)

        if self.verbose:
            print("Successfully extracted names of active WGs existing in files",
                  end= "\n------------------------------------------------------------\n")
            
        return active_wg_files

    
    def combine_wg_files(self, ratio: float=0.15) -> Type[pd.DataFrame]:
        """
        A method that transforms .csv files of all scraped
        active WGs into Pandas dataframes, ads WG name 
        in a new column and concatenates all dfs into one
        big df. 
        """
        
        active_wgs = self._extract_wgs()
        all_csv_files = os.listdir(self.archive_path)  # gathering all the file names
        filenames = [doc.rsplit(".", 1) for doc in all_csv_files]  # separating extension and filename
        active_wgs_in_files = self._active_groups_in_files(filenames, active_wgs) # list of active WGs present in files

        # Converting active WGs .csv file to Pandas dataframe, and storing the respective WG name
        active_dfs = [(pd.read_csv(os.path.join(self.archive_path, act_wg_file)), act_wg_file) for act_wg_file in active_wgs_in_files]

        # If ratio is given, sample random dataframes and their respective names
        if ratio:
            active_dfs = [(df.sample(frac=ratio, replace=False), act_wg_file) for df, act_wg_file in active_dfs]

        # Creating a new column in dfs to store the WG name
        for i, df_wg_name in enumerate(active_dfs):
            name_wo_extension = df_wg_name[-1].rsplit(".")[0]
            active_dfs[i][0]["Working Group"] = name_wo_extension      

        # Concatenating dataframes into one big dataframe
        big_df = active_dfs[0][0]
        for active_df, _ in active_dfs:
            big_df = pd.concat([big_df, active_df])

        if self.verbose:
            print("Successfully converted and concatenated all .csv files into one dataframe",
                  end="\n-------------------------------------------------------------------------\n")
        self.active_wg_dataframes = big_df

    
    def get_combined_wg_dataframes(self) -> Type[pd.DataFrame]:
        """
        Returns the dataframe of all scraped active WGs
        """
        return self.active_wg_dataframes

    
    def process_email_bodies(self, 
                             bodies_wgs: List[List[str]],
                             punc: bool=True, 
                             digits: bool=True, 
                             lower: bool=True,
                             newline: bool=True, 
                             n_encryp_lines: int=3, 
                             max_greeting_length: int=4,
                             date_line_depth: int=3) -> Tuple[List[List[str]], dict[str, int]]:

        """
        A function that performs extensive pre-processing of email bodies.
        
        1. First, it checks whether the body is quopri-encoded.
        In case encoding is unidentifiable, the body is removed. 
        
        2. Then it checks whether the body is ill-formated. In case it is,
        it gets removed. 
        
        3. Then it checks whether the body is encrypted. In case it is,
        it gets removed.

        4. Then it removes lines that contain salutations, lines of the format 
        'ON [DATE] AT [TIMESTAMP] [PERSON] WROTE' (which are parts of the original message).

        5. It also checks whether the body is an announcement. In case it is,
        it gets removed.

        6. It then removes lines containing farewells, as well as everything beneath them. 
        It also removes original message lines prepended with '>', as well as extracting only
        relevant text in case a certain body pattern is detected.

        7. Finally, it performs minor cleaning by removing links, substituting \n or \t 
        characters with whitespace, removing certain punctuation characters, and lowering the text.
        """

        processed_bodies = []   # Container for processed bodies

        removed_stats_dict = {'unknown_enc': 0,
                              'ill_from_format': 0,
                              'encryp': 0,
                              'announ': 0,
                              'empty_post_proc': 0,
                              'diverse_noise': 0,
                              'diff_lang': 0}

        # Declaring all the regexes. Longer ones are compiled over several lines

        farewells_regex = re.compile(
            '(?:\s+)?(?:[Tt]hank(?:s(?:\s+again)?|\s+[Yy]ou(?:\s+again)?)|[Mm]any\s+thanks|'
            '(?:[Bb]est|[Kk]ind)\s+[Rr]egards|[Yy]ours(?:\s*[Ii]rrespectively)?|'
            "[Bb]est|[Cc](?:heers|iao)|[Rr]'s|[Rr]egards|[Aa]ll\s+the\s+best)(?:,|\.|!|\s+)?|"
            '(?:\s+)?(?:-{1,2}(?:\s*[A-Z]?[a-z]+)?|(?:[A-Z][a-z]+(?:\s+[A-Z](?:[a-z]+)?)?)|[A-Z]{2})(?:\s+)?'
        )

        announce_regex = re.compile(
            '(?:[Aa] new Request for Comments is now available in online)|'
            '(?:[Aa] New Internet-Draft is available from the on-line)|'
            '(?:[A-Za-z]+ [A-Za-z]+ has requested publication)|'
            '(?:[Aa] new meeting session request has just been submitted)|'
            '(?:is inviting you to a scheduled [A-Za-z]+ meeting)|'
            '(?:[Rr]eviewer:\s+[A-Za-z]+(?:\s+[A-Za-z]+\s*)*)|'
            '(?:[Tt]he [A-Za-z]+ has received a request from the)|'
            '(?:[Tt]he following [A-Za-z]+ report has been .+ for)'
            '(?:[Tt]he following errata report)|'
            '(?:[Ee]vents without label \"editorial\")|'
            '(?:[Tt]he session\(s\) that you have requested)|'
            '(?:[Aa]n IPR disclosure that pertains to your Internet-Draft)|'
            '(?:\w+(?: \w+)? has entered the following ballot position)|'
            '(?:.+ working group (?:(?:changed the .+)|(?:is inviting you to a scheduled .+)) meeting)|'
            '(?:.+ working group .+ will hold a)'
        )

        dates_regex = re.compile(
            '^(?:[Oo]n\s+)(?:(?:[A-Za-z]{3},\s+[A-Za-z]{3}\s+\d{2},)|'
            '(?:[A-Za-z]{3}\s+\d{2},\s+\d{4},)|(?:\d{2}\/\d{2}\/\d{4}(?:\s+\d{2}:\d{2})?)|'
            '(?:\d{1,2}\/\d{1,2}\/\d{2})|(?:[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})|'
            '(?:[A-Za-z]+\s+\d{1,2},\s+\d{4})|(?:\d{4}-\d{2}-\d{2},)|'
            '(?:[A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})|)'
        )

        diverse_noise_regex = re.compile('(?:<font face)|'
                                         '(?:<mb>.+\/mb>)|^(?:pull requests *)|'
                                         '(?:<.+> was just expired)|'
                                         '^(?:<html)', flags=re.IGNORECASE)
        
        email_wrote_regex = '(?:<.+>\s+(?:(?:wrote)|(?:writes)))'

        diff_lang_regex = '(?:(?:(?:um|in)(?:.+)? )?nachricht)|(?:am um.+ sc)|(?:el.+ escribi)|(?: ha scritto)|(?:le.+ a crit)'

        ill_format_regex = '^\s*[Ff]rom:'
        
        base64_regex = '^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}={2})$'
        
        names_greetings_regex = "(?:^[A-Z][a-z]+,$)|^[Hh](?:ello|i|ey)|^[Dd]ear|^[Tt]hank(?:s|\s+you)"

        for body, wg in tqdm(bodies_wgs):
            # Declaring all the necessary variables to control the flow beneath
            num_encrypted_lines = 0
            updated_body_lines = []
            body_removed = False

            # Remove non-ascii characters
            body = body.encode('ascii', 'ignore').decode()

            # Decode quopri-encoding (utf-8 and windows-1252 are the most common)
            # Try to decode with 'utf-8' encoding 
            try:
                body = quopri.decodestring(body).decode('utf-8')
            except:
                # Try to decode with 'Windows-1252' encoding
                try:
                    body = quopri.decodestring(body).decode('windows-1252')
                    # In case none of the encodings above worked, simply remove the body
                except:
                    removed_stats_dict['unknown_enc'] += 1
                    continue

            # Running through EmailParser first
            body = EmailReplyParser(languages=self.lng).parse_reply(text=body)

            # Detecting diverse noise
            if re.search(diverse_noise_regex, body.lower()):
                removed_stats_dict['diverse_noise'] += 1
                continue

            # Bodies containg non-english words are hard to deal with and thus removed
            if re.search(diff_lang_regex, body.lower()):
                removed_stats_dict['diff_lang'] += 1
                continue

            # Remove carriage return 
            body = body.replace("\r", "")

            # Replace multiple new lines with only 1 new line
            body_wo_new = re.sub('\n{2,}', '\n', body)

            # Split body into lines
            body_lines = body_wo_new.split("\n")
            
            # Some messages start with 'From: ...', without being prepended with
            # '>' char. This essentially makes it hard to retrieve latest reply,
            # separated from the original message. Thus, the messages are considered
            # ill-formated and removed
            ill_format_result = re.match(ill_format_regex, body_lines[0])
            if ill_format_result:
                removed_stats_dict['ill_from_format'] += 1
                continue
            
            # Check whether the message body line is encrypted
            for line in body_lines:
                encrypted_line = re.fullmatch(base64_regex, line) 
                
                # Corner case: message is encrypted and is only 1 line long
                if encrypted_line and len(body_lines) == 1:
                    removed_stats_dict['encryp'] += 1
                    body_removed = True  # setting a flag to avoid further processing
                    break
                    
                elif encrypted_line:
                    num_encrypted_lines += 1

                else:
                    break
                    
                # If the threshold of encrypted lines is reached, remove the whole body,
                # since it is likely to be encoded
                if num_encrypted_lines == n_encryp_lines:
                    removed_stats_dict['encryp'] += 1
                    body_removed = True
                    break
    
            # Match-object for announcements
            announce_line = re.match(announce_regex, body_lines[0]) 

            # If announcement-body is detected, remove it
            if announce_line:
                removed_stats_dict['announ'] += 1
                body_removed = True 

            # Message is either an announcement or encryption, skip the rest of the processing 
            if body_removed:
                continue
            
            # Match object for greetings
            greetings_line = re.match(names_greetings_regex, body_lines[0])  
           
            # In case a pattern for a salutation is detected in the first line, and
            # the number of words in line <= the given max. greeting length, remove it.
            # The intuition is that lines with few words are more likely to contain a greeting
            if greetings_line and len(body_lines[0].split()) <= max_greeting_length:
                body_lines[0] = ''

            # Check whether first date_line_depth lines contain dates or
            # '[PERSON] WROTE'-like patterns, and remove if detected.
            # The higher the date_line_depth value, the more the accuracy, but
            # also the complexity. Default value of 4 seems to work well.
            first_n_lines = body_lines[:date_line_depth] 
        
            for i, line in enumerate(first_n_lines):
                # Match-objects
                date_line = re.search(dates_regex, line)
                wrote_pattern_line = re.search(email_wrote_regex, line)

                if date_line:
                    body_lines[i] = ''

                elif wrote_pattern_line:
                    body_lines[i] = ''

            # Go through every line
            for i, line in enumerate(body_lines):

                # If line is an empty string or starts wiht >, ignore it  
                orig_message_line = re.match('^\s*>', line) 
                if not line or orig_message_line:
                    continue
                     
                # Check for farewells, if present: ignore it & everything that follows.
                # Everything that follows farewells is usually not relevant.
                farewell_line = re.fullmatch(farewells_regex, line) 
                
                if not farewell_line:
                    updated_body_lines.append(line)
                    
                else:
                    break

                # This part of code is specifically tailored to extract right
                # body content from a mail specific to IETF, containing both the
                # written message and automatically generated text
                if 'COMMENT:' in line and i != len(body_lines) - 1:
                    if '-' * 10 in body_lines[i + 1]:  # 10 is arbitrary, but works well
                            updated_body_lines = body_lines[i + 2:]
                            break
            
            # Restoring the body to its original structure
            processed_body = "\n".join(line for line in updated_body_lines)
            
            # Remove links
            processed_body = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '[LINK]', processed_body)

            # Expand contractions
            body = contractions.fix(body)

            # Remove line-terminators
            if newline:
                processed_body = processed_body.replace("\n", " ")
                processed_body = processed_body.replace("\t", " ")

            # Replacing 2+ whitespaces with 1 whitespace
            processed_body = re.sub('\s{2,}', ' ', processed_body)
            
            # Remove punctuations
            if punc:
                processed_body = re.sub(r'r[^\w\s]', '', processed_body)
            
            # If punctuation is preserved, some specfifc pattern removal still has to take place
            elif not punc:
                processed_body = re.sub('(?!\w+)-{2,}(?!\w+)', '-', processed_body)  # word--word --> word-word
                processed_body = re.sub('[=\-_\+]{2,}', '', processed_body)  # == / -- / __ / ++ --> = / - / _ / +
                processed_body = re.sub('-?(?:\+\-){2,}(?:[\+\-])?', '', processed_body) # -+-+ / +-+- --> ''
                processed_body = re.sub('(?:[\*\.\,] ?){2,}', '', processed_body)  # .. / ** / . . / * * --> ''
                processed_body = re.sub('(?:[\\\/] ?){2,}', '', processed_body)  # sequences of slashes
                processed_body = re.sub('[~\|]', '', processed_body)  # removing chars. that are rarely used

            # Post-processing block. This requires a for-loop check, since
            # regular expressions end up with catastrophic backtracking, resulting
            # in an extremely slow code
            patterns = ['said: ', 'wrote: ', 'writes: ', '<.*>: ']
            for pattern in patterns:
                match = re.search(pattern, processed_body)
                if match:
                    processed_body = processed_body[match.end() - 1:]

            # Remove digits   
            if digits:
                processed_body = re.sub(r"\d+", "", processed_body)

             # After most of the processing, some bodies might be empty - skip if so
            if processed_body == '':
                removed_stats_dict['empty_post_proc'] += 1
                continue
            
            # Remove extra spaces and newlines from edges
            processed_body = processed_body.strip()

            # Lowercase
            if lower:
                processed_body = processed_body.lower()

            processed_bodies.append([processed_body, wg])

        return processed_bodies, removed_stats_dict