import os
import json
import tqdm


def extract_text(filename,output_as_list=False):
    '''
    Extract the text from the file name (json file) and
    index the content from paper_id, title, abstract and body_text fields
    Retuns - if output_as_list = True
                list of all the above values,
             else
                text of title, abstract and bodt_text
    '''
    file = open(filename)
    body_text = ""
    abstract = ""
    title = ""
    paper_id = ""

    paper_content = json.load(file, encoding='utf-8')

    #get the paper_id
    if 'paper_id' in paper_content:
        paper_id = paper_content['paper_id']
    #get the title, if available
    if 'title' in paper_content['metadata']:
        title = paper_content['metadata']['title']
    #get abstract.text, if available
    if 'abstract' in paper_content:
        for abs in paper_content['abstract']:
            abstract = abstract + abs['text']
    if 'body_text' in paper_content:
        for bt in paper_content['body_text']:
            body_text = body_text + bt['text']


    if output_as_list:
            return [paper_id,title.lower(),abstract.lower(),body_text.lower()]
    else:
        return (title + ' ' + abstract + ' ' + body_text + ' ').encode('utf-8')


if __name__ == '__main__':
    # creating a text file upfront rather that creating a string saves us a lot of RAM allocation and is multiple times
    # faster than the latter method
    f = open("data/initial_corpus.txt", "wb")
    for i,filename in tqdm(enumerate(os.listdir('data/pdf_json/'))):
        if filename.endswith(".json"):
            file_name = os.path.join('data/pdf_json/', filename)
            f.write(extract_text(file_name))
            f.write('\n'.encode('utf-8'))
        else:
            continue
    f.close()
    print('Finished Fetching Text data')
    print('{} documents fetched'.format(i))