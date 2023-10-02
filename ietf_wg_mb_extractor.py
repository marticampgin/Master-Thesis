import requests
import os
import pandas as pd
import quopri
import re
import contractions

from tqdm import tqdm
from mailparser_reply import EmailReplyParser
from bs4 import BeautifulSoup


class IETF_WG_MB_Extractor:
    """
    IETF WORKING GROUP MAIL BODY EXTRACTOR
    --------------------------------------
    
    This class scrapes the names of currentøy active IETF
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
    def __init__(self, verbose=True, archive_path="email-archives/"):
        self.wgs_url = r"https://datatracker.ietf.org/wg/#INT"
        self.archive_path = archive_path
        self.active_wg_dataframes = None
        self.verbose = verbose
        self.lng = ["en", "de", "fr"]

    def extract_wgs(self):
        """
        A function that scrapes all the active Working
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

    
    def active_groups_in_files(self, filenames, wgs):
        """
        A function that compares all the scraped
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

    
    def combine_wg_files(self, ratio=0.15):
        """
        A function that transforms .csv files of all scraped
        active WGs into Pandas dataframes, ads WG name 
        in a new column and concatenates all dfs into one
        big df. 
        """
        
        active_wgs = self.extract_wgs()
        all_csv_files = os.listdir(self.archive_path)  # gathering all the file names
        filenames = [doc.rsplit(".", 1) for doc in all_csv_files]  # separating extension and filename
        active_wgs_in_files = self.active_groups_in_files(filenames, active_wgs) # list of active WGs present in files

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

    
    def get_combined_wg_dataframes(self):
        """
        Returns the dataframe of all scraped active WGs
        """
        return self.active_wg_dataframes

    
    def process_email_bodies(self, 
                             bodies_wgs,
                             punc=True, 
                             digits=True, 
                             lower=True,
                             newline=True, 
                             n_encryp_lines=3,
                             n_equal_signs=3,
                             qp_threshold=8, 
                             max_greeting_length=4,
                             date_line_depth=3):

        """
        A function that performs extensive pre-processing of email bodies.
        
        1. First, it checks whether the body is quoted-printable encoded.
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

        # Container for processed bodies
        processed_bodies = []
        removed_stats_dict = {'unknown_enc': 0,
                                  'ill_format': 0,
                                  'encryp': 0,
                                  'announ': 0}

        # Declaring all the regexes. Longer ones are compiled over several lines
        ill_format_regex = '^\s*[Ff]rom:'
        
        base64_regex = '^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}={2})$'
        
        names_greetings_regex = "(?:^[A-Z][a-z]+,$)|^[Hh](?:ello|i|ey)|^[Dd]ear|^[Tt]hank(?:s|\s+you)"
        
        farewells_regex = re.compile(
            '(?:\s+)?(?:[Tt]hank(?:s(?:\s+again)?|\s+[Yy]ou(?:\s+again)?)|[Mm]any\s+thanks|'
            '(?:[Bb]est|[Kk]ind)\s+[Rr]egards|[Yy]ours(?:\s*[Ii]rrespectively)?|'
            "[Bb]est|[Cc](?:heers|iao)|[Rr]'s|[Rr]egards|[Aa]ll\s+the\s+best)(?:,|\.|!|\s+)?|"
            '(?:\s+)?(?:-{1,2}(?:\s*[A-Z]?[a-z]+)?|(?:[A-Z][a-z]+(?:\s+[A-Z](?:[a-z]+)?)?)|[A-Z]{2})(?:\s+)?'
        )

        announce_regex = re.compile(
            '(?:A new Request for Comments is now available in online)|'
            '(?:A New Internet-Draft is available from the on-line)|'
            '(?:[A-Za-z]+ [A-Za-z]+ has requested publication)|'
            '(?:A new meeting session request has just been submitted)|'
            '(?:is inviting you to a scheduled [A-Za-z]+ meeting)|'
            '(?:Reviewer:\s+[A-Za-z]+(?:\s+[A-Za-z]+\s*)*)|'
            '(?:The [A-Za-z]+ has received a request from the)|'
            '(?:The following [A-Za-z]+ report has been submitted for)'
        )

        dates_regex = re.compile(
            '^(?:[Oo]n\s+)(?:(?:[A-Za-z]{3},\s+[A-Za-z]{3}\s+\d{2},)|'
            '(?:[A-Za-z]{3}\s+\d{2},\s+\d{4},)|(?:\d{2}\/\d{2}\/\d{4}(?:\s+\d{2}:\d{2})?)|'
            '(?:\d{1,2}\/\d{1,2}\/\d{2})|(?:[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})|'
            '(?:[A-Za-z]+\s+\d{1,2},\s+\d{4})|(?:\d{4}-\d{2}-\d{2},)|'
            '(?:[A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})|)'
        )

        email_wrote_regex = re.compile(
            '(?:<.+>\s+(?:(?:wrote)|(?:writes)))'
        )


        for body, wg in tqdm(bodies_wgs):
            # Declaring all the necessary variables to 
            # control the flow beneath
            num_encrypted_lines = 0
            updated_body_lines = []
            body_removed = False
            
            # Detect quoted-printable encoding by looking for '=0A=' char sequence.
            # Usually, the text is encoded using either utf-8 or windows-1252 encodings.
            if '=0A=' in body:
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

            # Not all quoted-printable-encoded messages include '=0A=' sequence.
            # Looking for '=' character at the end of lines
            else:
                temp_lines = body.split('\n')  # splitting lines by \n char
                eq_lines_counter = 0
                iters_since_last_hit = 0
                
                for line in temp_lines:
                    # If more than qp_threshold lines didn't finish with '=' since last hit,
                    # then it is likely that the message is not quoted-printable-encoded and
                    # we stop iterating further
                    if iters_since_last_hit >= qp_threshold:
                        break

                    # If the line is not empty and ends with '='
                    if line and line[-1] == '=':
                        eq_lines_counter += 1  # got a hit
                        iters_since_last_hit = 0  # reset iterations since last hit

                        # If we reach the desired amount of lines that end with '='
                        # and streak haven't been broken by iters_since_last_hit, it is likely that
                        # message is quoted-printable-encoded.
                        if eq_lines_counter == n_equal_signs:
                            try:
                                body = quopri.decodestring(body).decode('utf-8')
                            except:
                                try:
                                    body = quopri.decodestring(body).decode('windows-1252')
                                # If none of the above encodings are detected, remove the body
                                except:
                                    removed_stats_dict['unknown_enc'] += 1
                                    body_removed = True
                                    break           
                    else:
                        # Only add iterations since the last time '=' was detected if
                        # '=' was detected in the first place
                        if eq_lines_counter > 0:
                            iters_since_last_hit += 1

            # In case body hasn't been removed, proceed with further processing
            if body_removed:
                continue
            else:
                # Remove carriage return 
                body = body.replace("\r", "")
                # Replace many new lines with only 1 new line
                body_wo_new = re.sub('\n{2,}', '\n', body)
                # Split body into lines
                body_lines = body_wo_new.split("\n")

                # Check if the message is ill-formatted, remove if so
                ill_format_result = re.match(ill_format_regex, body_lines[0])

                if ill_format_result:
                    removed_stats_dict['ill_format'] += 1
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
    
                # Message is encrypted, skip the rest of the processing
                if body_removed:
                    continue

                # Potential Match-objects for salutations and announcements
                greetings_line = re.match(names_greetings_regex, body_lines[0])  
                announce_line = re.search(announce_regex, body_lines[0]) 
                
                # In case a pattern for a salutation is detected in the first line, and
                # the number of words in line <= the given max. greeting length, remove it.
                # The intuition is that lines with few words are more likely to contain greeting
                if greetings_line and len(body_lines[0].split()) <= max_greeting_length:
                    body_lines[0] = ''

                # Check whether first date_line_depth lines contain dates or
                # '[PERSON] WROTE patterns', and remove if detected.
                # The higher the date_line_depth value, the more the accuracy, but
                # also the complexity
                first_n_lines = body_lines[:date_line_depth] 
            
                for i, line in enumerate(first_n_lines):
                    # Match-objects
                    date_line = re.search(dates_regex, line)
                    wrote_pattern_line = re.search(email_wrote_regex, line)

                    if date_line:
                        body_lines[i] = ''

                    elif wrote_pattern_line:
                        body_lines[i] = ''
                        
                # If announcement-body is detected, remove it
                if announce_line:
                    removed_stats_dict['announ'] += 1
                    body_removed = True 
                        
                # Don't perform any processing if the body was removed
                if body_removed:
                    continue
                    
                else:
                    # Go through every line
                    for i, line in enumerate(body_lines):

                        # Expand contractions
                        line = contractions.fix(line)    
                        
                        # If line is an empty string, ignore it
                        if not line:
                            continue
                            
                        # If line starts with >, ignore it 
                        orig_message_line = re.match('^\s*>', line) 

                        if orig_message_line:
                            continue
                    
                        # Check for farewells, if present: ignore it & everything that follows it
                        farewell_line = re.fullmatch(farewells_regex, line) 
                        
                        if not farewell_line:
                            updated_body_lines.append(line)
                            
                        else:
                            break
        
                        # This part of code is specifically tailored to extract the mail 
                        # from a a mail specific to IETF, containing both the mail and
                        # automatically generated text
                        if 'COMMENT:' in line and i != len(body_lines) - 1:
                            if '-' * 10 in body_lines[i + 1]:
                                 updated_body_lines = body_lines[i + 2:]
                                 break
            
            # Restoring the body to its original structure
            processed_body = "\n".join(line for line in updated_body_lines)
            
            # Running through EmailParser to further strip any original messages that
            # weren't removed with the processing above.
            processed_body = EmailReplyParser(languages=self.lng).parse_reply(text=processed_body)
            
            
            # Remove links
            # FTP links? ftp://ftp.ietf.org/internet-drafts/
            # (https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)

            processed_body = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '', processed_body)
            

            # Remove punctuations (except dots and commas, for now at least)
            
            '''
            BEFORE REMOVING PUNCTUATION WE COULD TRY AND CREATE REGEX TO DETECT
            SMILES/EMOTICONS AND KEEP THEIR INITIAL COORDINATES, 
            AND THEN ADD THEM, OR SMTH LIKE THAT
            '''

            '''
            FOR TEXT ANALYTICS, WE WILL REMOVE PUNCTUATION AND KEEP ONLY THE WORDS.
            FOR DEEP-LEARNING, WE WILL KEEP MOST (HAVE TO DECIDE HOW MUCH) OF THE PUNCTUATION.
            '''

            if punc:
                # r'[^\w\s\.\,\?\!]' - keeps more of the context
                processed_body = re.sub(r'[^\w\s]', '', processed_body)
                
            # Remove all digits
            if digits:
                processed_body = re.sub(r"\d+", "", processed_body)
                
            # Remove extra spaces and newlines
            processed_body = processed_body.strip()
            
            if newline:
                processed_body = processed_body.replace("\n", " ")
                processed_body = processed_body.replace("\t", " ")
            
            # Lowercase
            if lower:
                processed_body = processed_body.lower()
                
            processed_bodies.append([processed_body, wg])

        return processed_bodies, removed_stats_dict