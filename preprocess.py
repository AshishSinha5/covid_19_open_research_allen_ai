from tqdm import tqdm
import re
import subprocess
import sys
import gc

import nltk
nltk.download('punkt')

try:
    import langid
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'langid'])
finally:
    import langid


try:
    import in_place
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'in_place'])
finally:
    import in_place


def remove_non_english_3(file_path):
    """
    args: file_path - file path of the corpus
    returns: removes all non english articles from the corpus as we are interested only in english articles
    """
    print('---------------------------------------------------------')
    print('Removing all non english samples')
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            if langid.classify(line)[0] == 'en':
                file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


def remove_brackets(file_path):
    """
    args: file_path - file path of the corpus
    returns: removes all text inside brackets e.g. here is (this will be deleted) an example. --> here is an example.
    since the text inside a bracket wont help in language modeling so we remove it
    """
    print('---------------------------------------------------------')
    print("Removing parentheis () and text inside it")
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'\((?:.*?)\)', '', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


def remove_space_url(file_path):
    """
    args: file_path - file path of the corpus
    returns: removes all extra spaces in text corpus also all the urls since they are not useful in language modeling
    """
    print('---------------------------------------------------------')
    print("Removing Spaces and URLs")
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'[(http:\/\/)|\w]*?[\w]*\.[-\/\w]*\.\w*[(\/{1})]?[#-\.\/\w]*[(\/{1,})]?', ' ', line)
            # line = re.sub(r'\s{2,}',' ', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')


def remove_cite_figs(file_path):
    """
    args: file_path - file path of the corpus
    returns: removing all ciatation and figure references
    """
    print('---------------------------------------------------------')
    print('Removing Citation and Figure References')
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'\[.*?\]', '', line)
            # replacing all figure references outside brackets with "figure"
            line = re.sub(r'\s+(?:Fig\.|Figure|fig\.|figure)\s*\d+\s*\.\s*\d*\s+', ' figure ', line)
            # removing all figure references inside parenthesis e.g. (Fig. 12.1)
            line = re.sub(r'\(\s*(?:Fig\.|Figure|fig\.|figure) \s*\d+\s*\.*\s*\d*\s\)', '', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


def replace_words(file_path, rep_dict):
    """
    args: file_path - file path of the corpus
          rep_dect - dictionary containing mapping between words
    returns: replacing frequetly occuring similar sounding words such as (Covid-19, covid-19, coronavirus, cirina, Sars-cov-2) --> (covid-19)
            also replacing units of measurement such as kg/m to kilogram per meter, etc.
            list is not exhaustive and was discovered while exploring the corpus
    """
    print('---------------------------------------------------------')
    print('Replacing Similar Sounding Words')
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            pat = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in rep_dict.keys()) + r')(?!\w)')
            line = pat.sub(lambda x: rep_dict[x.group()], line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


def remove_special_char(file_path):
    """
    args : file_path - file_path of corpus
    returns:
    removes all special charecters from the corpus except "." handles corner cases like - and / which may either be used in hyphenated words
    for relaying 'or' statements.
    We pad these to special charencter with space before removing also remove \n at the end of the text and then proceed to remove remaining
    charecters
    """
    print('---------------------------------------------------------')
    print('Removing Special Charecters')
    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'(?<=\W)\/(?=\W+)', ' ', line)
            line = re.sub(r'(?<=\W)\-(?=\W+)', ' ', line)
            # line = re.sub(r'\\n',"", line)
            # line = re.sub(r'\s\s+', " ", line)
            line = re.sub(r'\/|\-|@|&|\'|\"|\(|\)|<|>|#|%|\^|\*|\$|;|:|\{|\}|\+|\=|\!|\\|,|\[|\]', "", line)
            line = re.sub(r' (?= |$)', '', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


def remove_num(file_path):
    """
    args : file_path - file_path of corpus
    returns:
    removes all standalone numbers and decimal numbers
    """
    print('---------------------------------------------------------')
    print('Removing standalone numbers and decimals')

    with in_place.InPlace(file_path, encoding='utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'(\b\d+(?:\.\d+)?(\s+))|(\b\d+(?=\.))', "", line)
            line = re.sub(r' (?= |$)', '', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    return


if __name__ == "__main__":
    file_path = 'data/initial_corpus.txt'
    replace_dict = {'COVID-19': 'covid-19',
                    'Covid-19': 'covid-19',
                    'covid-19': 'covid-19',
                    'corona': 'covid-19',
                    'coronavirus': 'covid-19',
                    'SARS-CoV-2': 'covid-19',
                    'mg/kg/day': 'milligrams per kilogram per day',
                    'ng/l': 'nanogrmas per liter',
                    'ng/dl': 'nanograms per deciliter',
                    'p/μL': 'parasite per microliter',
                    'pcg/ml': 'picogram per milliliter',
                    'µg/mL': 'microgram per milliliter',
                    'mg/ml': 'milligram per milliliter',
                    'sec/mm2': 'second per millimeter squared',
                    'mOsm/liter': 'particles per liter',
                    'pg/µL': 'picogram per milliliter',
                    'mg/mL': 'milligram per milliliter',
                    'mmHg/ml': 'millimetre of mercury per milliliter',
                    'miles/day': 'miles per day',
                    'cc/kg': 'cubic centimeters per kilogram',
                    'mg/L': 'milligram per liter',
                    'mL/kg/hour': 'milliliter per kilogram per hour', }

    remove_non_english_3(file_path)
    remove_brackets(file_path)
    remove_space_url(file_path)
    remove_cite_figs(file_path)
    replace_words(file_path, replace_dict)
    remove_special_char(file_path)
    remove_num(file_path)