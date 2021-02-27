from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re

for subset in ['afe', 'apw', 'xie']:
    print(subset)
    for file_name in tqdm(os.listdir('./gigaword_eng/{}/'.format(subset))):
        file_path = os.path.join('./gigaword_eng/{}/'.format(subset), file_name)
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
            with open('./gigaword_txt/{}/'.format(subset)+file_name+'.txt', 'w') as file_2:
                for paragraph in external_paragraphs:
                    file_2.write('{}\n'.format(paragraph))

print('boom!')