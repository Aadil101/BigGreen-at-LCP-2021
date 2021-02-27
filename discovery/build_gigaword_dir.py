from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import os
import re

def main(pwd):
    for subset in ['afe', 'apw', 'nyt', 'xie']:
        print(subset)
        if not os.path.isdir('{}/gigaword_txt/{}'.format(pwd, subset)):
            os.mkdir('{}/gigaword_txt/{}'.format(pwd, subset))
        for file_name in tqdm(os.listdir('{}/gigaword_eng/{}/'.format(pwd, subset))):
            file_path = os.path.join('{}/gigaword_eng/{}/'.format(pwd, subset), file_name)
            with open(file_path, 'r') as file_1:
                external_paragraphs = []
                soup = BeautifulSoup(file_1, "html.parser")
                # Iterate over all <p> items and get the text for each.
                for paragraph in soup("p"):
                    # Turn inter-paragraph newlines into spaces
                    paragraph = paragraph.get_text()
                    paragraph = re.sub(r"\n+", "\n", paragraph)
                    paragraph = paragraph.replace("\n", " ")
                    external_paragraphs.append(paragraph)
                with open('{}/gigaword_txt/{}/'.format(pwd, subset)+file_name+'.txt', 'w') as file_2:
                    for paragraph in external_paragraphs:
                        file_2.write('{}\n'.format(paragraph))
    print('boom!')

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    # Add the arguments
    my_parser.add_argument('--pwd',
                        type=str,
                        help='pwd of Gigaword-5')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    pwd = args.pwd
    main(pwd)