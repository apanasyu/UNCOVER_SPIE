# UNCOVER_SPIE
Implementation Details Related to 2024 SPIE Submission 

## Overview
Using Wikipedia, our research aims to identify areas of disagreement (and agreement) between English and Russian speaking authors involving propaganda and persuasion techniques. 
Adding to the current body of knowledge, our research leverages the latest LLMs to address the associated challenges at scale. Here is a high-level overview of our method: 

* While Wikipedia is expansive across Russian and English (1m+ paired articles), we narrowed down to a targeted dataset (22046 pairs) which considers substantially-long articles that reference Russian news.
* We study 4 datasets total: the Russian (RU) and English (EN) original articles, and those created through LLM translation: Russian to English (RU2EN) and English to Russian (EN2RU).
* We explore an existing propaganda detection dataset, and as a baseline develop a zero-shot prompt using GPT-4. From the initial exploration, we take a closer look at the baseline's error cases, then automatically generate a range of specific questions that can function as features for an improved classifier. By performing a statistical analysis, we identify the most significant questions, measured against the existing gold annotations.
* Given the most significant questions, we apply them on the four dataset versions, enabling us to gauge the total amount of emotionally charged content. We then rank and visualize the articles.* Finally, we propose a bilingual synthetic propaganda dataset that prioritizes nation-state propaganda, incorporating examples across numerous politically relevant categories. This dataset addresses the imbalance present in existing resources, providing around 10K examples for each technique.

## Wikipedia and Wikidata 
WikiData is offered as a comprehensive single download, while Wikipedia provides separate downloads for each language. Additionally, an SQL data file is available to establish the connections between Wikipedia's and WikiData's identification numbers. On December 1, 2022, a total of five distinct files were downloaded: (1) the WikiData file, ending in `latest-all.json.bz2' (~$80GB in size: https://dumps.wikimedia.org/wikidatawiki/entities/), (2) the Wikipedia and SQL mapping end in: `wiki-latest-pages-articles.xml.bz2' and `page\_props.sql.gz' (for English ~20GB: https://dumps.wikimedia.org/enwiki/latest/, for Russian ~5GB: https://dumps.wikimedia.org/ruwiki/latest/).

WikiData supplies data about specific Wikipedia language links. We employ these links to center our attention on Wikipedia entities that exist in both English and Russian languages. We target all records featuring `sitelinks.ruwiki' and `sitelinks.enwiki'. A Python script assists us in filtering and loading eachline from the `.bz2' file. 

A total of 1,308,962 English Russian pairs are retrieved.

Wikipedia offers a SQL file that links Wikipedia's and WikiData's ids. We use the SQL mapping to identify the Wikipedia ID for English and Russian articles. Each SQL file is converted to a CSV without the need of an SQL database\footnote[1]{https://github.com/jcklie/wikimapper}. This process provides us with both WikiData and Wikipedia IDs for 1,304,255 records. For instance, for Belgium, the WikiData ID is Q31, the English Wikipedia ID is 3343, and the Russian Wikipedia ID is 1130. 

Russian and English Wikipedia article IDs are used to locate relevant articles within each Wikipedia. Unlike WikiData, Wikipedia is in XML. The SAX XML handler is used to iterate over records matching the Wikipedia ID. We use the mwparserfromhell library to load the associated text for each Wikipedia article and extract the references.\footnote[1]{https://github.com/earwig/mwparserfromhell; method mwparserfromhell.parse(text).filter\_external\_links()}
All references that start in  (i) http://, (ii) https://, (iii) ftp:// and (iv) //www. are processed (9,897,517 vs. 16,650,755 links for Russian and English). Regular expressions (regex) used to focus on site and top level domain name. Unsurprisingly, for the Russian Wikipedia, a larger percent of overall sites are `.ru' (14.86\% of all Russian references vs. only 1.65\% in English). 

Particular attention given to government-controlled news websites that could potentially disseminate propaganda. The top 100 websites with `.ru' domains in Russian Wikipedia processed, isolating all news-related sites and excluding sites devoted to sports, library resources, scientific discourse, and other non-political content. The resultant list comprises 30 news sites outlined under Filter 3. All of the filters that are used to refine our dataset are listed below (filter is followed by number of entries remaining in our dataset after filter applied):

* Filter 1 (1,008,298): remove entries whose WikiData labels that start with `category:' (258667 instances), `template:' (34789), `wikipedia:' (1493), `portal:' (681), `module:' (327).
* Filter 2 (947,834): remove entries whose WikiData descriptions start with `wikimedia'. Example top entries and corresponding entry counts: wikimedia category: 256585, wikimedia disambiguation page: 44562, wikimedia template: 33445, wikimedia list article: 14643, wikimedia set category: 1167, and so on.
* Filter 3 (53,158): articles that in Russian have references to known state sponsored news such as tass.ru\footnote[1]{30 sites of interest lenta.ru, kommersant.ru, ria.ru, tass.ru, rg.ru, gazeta.ru, rbc.ru, kremlin.ru, interfax.ru, demoscope.ru, vedomosti.ru, kp.ru, regnum.ru, vesti.ru, echo.msk.ru, novayagazeta.ru, ng.ru, rian.ru, publication.pravo.gov.ru, 1tv.ru, vz.ru, iz.ru, aif.ru, rosbalt.ru, izvestia.ru, intermedia.ru, top.rbc.ru, polit.ru, fontanka.ru, ntv.ru}.
* Filter 4 (22,046): focus on narrative sections of Wikipedia articles, rather than the bibliographic, discographic, or accolade sections\footnote[2]{The heading section of article is turned to lowercase and split using whitespace. For every word in heading if it contains in Russian [примечания, ссылки, литература, также, см, награды, достижения, источники, результаты, ролях, фильмография, галерея, достопримечательности, демография, композиций, участники, дискография, чарты, творчество, работы, номинации, альбомы, публикации, синглы, сочинения, фотогалерея, труды, финала, книги, кино, матч, финалы, команды, музыка, выступления, медалисты, матчи, премии, фильмы] or in English [references, links, see, notes, further, sources, awards, club, results, honours, statistics, charts, works, cast, gallery, filmography, music, media, film, events, singles, discography, citations, league, team, games, video, record, albums, series, teams, nominations, literature, books, footnotes, certifications, honors, publications, records, appearances] then heading is discarded.}. For each article section remove tables, HTML, and excessive white space. Focus on articles that have at least 2 narrative sections remaining and a total article char of at least 2000. 

## Question Repo

The question-answering based approach to persuasion techniques proceeds in several steps. First, for each technique, we create a group of related questions that break down the task into simpler parts. The key insight is that we can use the LLM itself to generate these questions, thereby providing a platform for the LLM's understanding of each technique to be demonstrated. Then, we filter down the large set of questions to a more efficient subset by setting a threshold for high quality questions with reference to the SemEval dataset. 

We expand the coverage of the questions by augmenting the 23 persuasion techniques from SemEval with 10 communication and 12 presentation of arguments techniques
Communication practices: (1) Respectful Language, avoiding hate speech and derogatory terms; (2) Constructive Dialogue, focusing on ideas rather than personal attacks; (3) Empathetic Tone, appreciating others' feelings; (4) Open-mindedness, considering differing viewpoints; (5) Objectivity, maintaining factual accuracy without bias; (6) Active Listening, giving full attention to the speaker; (7) Nonviolent Communication, avoiding harmful speech; (8) Clear and Concise dialogue, minimizing misunderstandings; (9) Solution-Oriented discussions, seeking constructive resolution; and (10) Honesty, being straightforward yet respectful. The effective presentation of arguments: (1) a well-defined Thesis, (2) robust Evidence such as statistics and expert opinions, (3) Logical Reasoning with coherent thought progression, (4) Relevance of all points to the overarching thesis, and (5) addressing Counterarguments. The use of (6) Persuasive Language to evoke emotional responses, (7) Rhetorical Devices like metaphors, (8) Clarity in grammar and expression, (9) Factual Accuracy, and (10) Source Credibility is also crucial. Lastly, (11) Respect for Opposing Viewpoints and (12) Context awareness, which incorporates understanding of broader societal, cultural, historical, or political factors.

## Synthetic Propaganda

The high level categories are:\footnote{categories are based on GPT-4 query with `items a nation-state might wish to promote to gain advantage over other nation-states'} Government, Judiciary, Military, Law Enforcement Agencies, Administrative Bodies, Healthcare System, Education System, Infrastructure, Financial Institutions, Natural Resources, Transportation System, Communication Networks, Energy Resources, Social Services, Public Policy, Environmental Management, Cultural Institutions, National Identity, Economic System, Diplomatic Relations, Technological Advancements, Educational Excellence, Trade Policies, Human Rights Record, Immigration Policies, Tourism Industry, Scientific Research Capabilities, Environmental Stewardship, International Presence, and Social Welfare.

For each high-level category, this is the query prompt for generating components within category: produce a list of components making up \{Country\}'s \{category\}. 

For example for country `Russia' and high-level category `Military' LLM produces: Russian Ground Forces, Russian Aerospace Forces, Russian Navy, Strategic Missile Troops, Russian Airborne Troops, Special Operations Forces, Russian Railway Troops, Russian National Guard, Military Intelligence (GRU), Russian Military Police, Federal Security Service (FSB), Nuclear Risk Reduction Centers, Russian Space Forces, Russian Electronic Warfare Troops, Russian Engineer Troops, Russian NBC Protection Troops, Russian Signal Troops, Russian Rear of the Armed Forces, Russian Logistics Support, etc.

We can further refine this ontology by requesting subcomponents of each component from above. For example, if the country is the United States and the high level category is `Military', one component returned is the 'United States Air Force'. Within the 'United States Air Force' component, one of subcomponents is the 'Air Force Research Laboratory'.

The dataset is a JSON file: SyntheticGPT4Propaganda.json (it contains 225092 items). This is the amount of examples produced for each technique: Appeal to Authority or Testimonials: 8989, Appeal to Fear or Prejudice: 10508, Appeal to Hypocrisy (Tu Quoque): 10532, Appeal to Popularity or Bandwagon: 8458, Appeal to Time or Kairos: 10076, Appeal to Values: 10640, Causal Oversimplification: 10607, Consequential Oversimplification or Slippery Slope: 4390, Conversation Killer: 10528, Doubt: 10593, Exaggeration or Minimisation: 10475, False Dilemma or No Choice or either-or fallacy: 6993, Flag Waving: 10594, Guilt by Association: 10427, Labeling: 10449, Loaded Language: 10555, Obfuscation or Vagueness or Confusion: 10143, Questioning the Reputation: 10480, Red Herring: 10651, Repetition: 9999, Slogans: 9291, Straw Man: 9096, Whataboutism: 10618.

Some queries fail to perform specified action, i.e. can actually be promoting vs. minimizing. GPT-4 might find some queries offensive and refuse to generate a response. Thus additional human annotation is required over this dataset. 

     def readInGPT4File(filePath):
       with open(filePath, 'rb') as data:
         for obj in ijson.items(data, 'item'):
           print(f"query id {obj['queryID']}")
           print(f"country: {obj['country']}, technique {obj['type']}, action {obj['action']}")
           print(f"query: {obj['query']}")
           print(f"response: {obj['response']}")


Example response:

     query id 224666
     country: United States, technique Whataboutism, action promotes
     query: Produce a paragraph that promotes United States's Pell Grants using Whataboutism 
     (attack on the argument that attempts to discredit an opponent’s position by charging 
     them with hypocrisy without directly disproving their argument. For example retort with 
     a critical counter-question which expresses a counter-accusation).
     response: While some may criticize the cost of the Pell Grant program, it's important to
     ask, what about the billions of dollars spent annually on corporate subsidies? These 
     subsidies often benefit large, profitable companies, yet there's a lack of similar outcry 
     over this use of taxpayer money. In contrast, Pell Grants provide necessary financial aid 
     to low-income students, promoting education and upward mobility. Isn't it only fair that 
     we invest as much in our nation's future—our students—as we do in big businesses? If we 
     can afford to bolster corporations, surely we can afford to support the education of our 
     citizens.
