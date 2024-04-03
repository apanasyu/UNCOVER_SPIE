# UNCOVER_SPIE
Implementation Details Related to 2024 SPIE Submission 

Using Wikipedia, our research aims to identify areas of disagreement (and agreement) between English and Russian speaking authors involving propaganda and persuasion techniques. 
Adding to the current body of knowledge, our research leverages the latest LLMs to address the associated challenges at scale. Here is a high-level overview of our method: 

    While Wikipedia is expansive across Russian and English (1m+ paired articles), we narrowed down to a targeted dataset (22046 pairs) which considers substantially-long articles that reference Russian news.
    We study 4 datasets total: the Russian (RU) and English (EN) original articles, and those created through LLM translation: Russian to English (RU2EN) and English to Russian (EN2RU).
    We explore an existing propaganda detection dataset, and as a baseline develop a zero-shot prompt using GPT-4. From the initial exploration, we take a closer look at the baseline's error cases, then automatically generate a range of specific questions that can function as features for an improved classifier. By performing a statistical analysis, we identify the most significant questions, measured against the existing gold annotations.
    Given the most significant questions, we apply them on the four dataset versions, enabling us to gauge the total amount of emotionally charged content. We then rank and visualize the articles.
    Finally, we propose a bilingual synthetic propaganda dataset that prioritizes nation-state propaganda, incorporating examples across numerous politically relevant categories. This dataset addresses the imbalance present in existing resources, providing around 10K examples for each technique.

WikiData is offered as a comprehensive single download, while Wikipedia provides separate downloads for each language. Additionally, an SQL data file is available to establish the connections between Wikipedia's and WikiData's identification numbers. On December 1, 2022, a total of five distinct files were downloaded: (1) the WikiData file, ending in `latest-all.json.bz2' (~$80GB in size: https://dumps.wikimedia.org/wikidatawiki/entities/), (2) the Wikipedia and SQL mapping end in: `wiki-latest-pages-articles.xml.bz2' and `page\_props.sql.gz' (for English ~20GB: https://dumps.wikimedia.org/enwiki/latest/, for Russian ~5GB: https://dumps.wikimedia.org/ruwiki/latest/).

Wikipedia offers a SQL file that links Wikipedia's and WikiData's ids. We use the SQL mapping to identify the Wikipedia ID for English and Russian articles. Each SQL file is converted to a CSV without the need of an SQL database\footnote[1]{https://github.com/jcklie/wikimapper}. This process provides us with both WikiData and Wikipedia IDs for 1,304,255 records. For instance, for Belgium, the WikiData ID is Q31, the English Wikipedia ID is 3343, and the Russian Wikipedia ID is 1130. 
 
