## Language Model of Covid-19 data by Allen Ai - (CORD-19)

The project is a part of ongoing **NLP** course at [Chennai Mathematical Institute](https://www.cmi.ac.in/teaching/msc-data-science/index.html).

#### Status: Active

## Objective
The purpose of this project is to build a **language model/pseudo research content** for COVID-19 texual data of over 55,000 documents. We can further go on to do knowledge discovery on the preprossesed corpus to answer questions such risk factors of covid disease, information about the vaccines, origin of virus, etc. Many of these questions are suitable for text mining and we aim to build text mining tools to provide insights on these questions.

## Dataset

We have used partial data set, sourced from Kaggle [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) contains research articles related to various specializations related to COVID-19.
Download the dataset from [here](https://drive.google.com/drive/folders/1f2pSuVT2cU8NGTY5c4mtPihgyIZYF__m) and directory structure would look as follows

<pre><code>
|--data
    |--json_directory
        --file_1.json
        --file_2.json
        --...
--preprocess.py
--...
</code></pre>

## Getting Started

1. Clone this repo to your local machine (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Add data files as explained above
3. Extract Relevant data from json - 
<pre><code>
python extract_data.py
</code></pre>
4. Preprocess the text corpus - 
<pre><code>
python preprocess_data.py
</pre></code>

## Preprocessing Steps

- Information extracted from Initial Corpus
  - paper ID
  - paper title
  - abstract text (only text field)
  - body text without bib references without section info
- Preprocessing Steps on the extracted corpus
  - remove foreign language articles
  - remove everything inside brackets
  - remove extra scpaces and urls
  - remove ciatations and figure references
  - replace words using replacement dictionary
  - remove all special characters except "."
  - remove standalone numbers and decimal numbers

## Initial Token Analysis
> Number of Sentences - 7644941 <br>
> Number of words  = 204224987 <br>
> Vocabulary Size = 2097602 <br>
- Heaps' law also seems to hold  for our preprocessed corpus

<p align="center">
  <img width="460" height="300" src="https://github.com/AshishSinha5/covid_19_open_research_allen_ai/blob/master/plots/heaps_law.png">
</p>
