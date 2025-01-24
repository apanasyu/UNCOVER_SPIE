import pandas as pd
import os

def getRepositoryOfQuestions(dirOut, fileNames, questionOutCSV):
    dataValues = []

    generateRepository = True
    repositoryOfQuestionsIndividual = []
    repositoryOfQuestionsGrouped = []
    questionCount = 0
    questionCountToQuestion = {}
    questionCountToQuestionCategory = {}
    queryGroups = []
    if generateRepository:
        for fileName in fileNames:
            repositoryOfQuestionsFile = []

            inFile = dirOut + fileName
            import re
            if os.path.isfile(inFile):
                with open(inFile, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("response: "):
                            line = line.strip()[10:]
                            # line = re.sub(' \\(.*?\\)', '', line)
                            if ";" in line:
                                items = line.split(";")

                                #questionCount = 0
                                questionGrouping = ""
                                for item in items:
                                    item = item.strip()
                                    if item.endswith("."):
                                        item = item[:-1]

                                    question = item
                                    if len(question) > 5 and question.endswith("?"):
                                        repositoryOfQuestionsIndividual.append(question)
                                        questionGrouping += f"Q{questionCount}: {question} "

                                        questionCountToQuestion[f"Q{questionCount}"] = question
                                        questionCountToQuestionCategory[f"Q{questionCount}"] = queryGroups[len(queryGroups)-1]
                                        row = [f"Q{questionCount}", queryGroups[len(queryGroups)-1], question]
                                        dataValues.append(tuple(row))

                                        repositoryOfQuestionsFile.append(question)
                                        questionCount += 1

                                repositoryOfQuestionsGrouped.append(questionGrouping.strip())
                        elif line.startswith("query: "):
                            line = line.strip()[len("query: "):]
                            line = line.split(": ")[0]
                            group = line[len("Given a piece of text your goal is to identify whether the text has "):]
                            queryGroups.append(group)

    df = pd.DataFrame(dataValues, columns=["Question ID", "Grouping", "Question"])
    df.to_csv(questionOutCSV, index=False)

    print(len(repositoryOfQuestionsGrouped))
    print(repositoryOfQuestionsGrouped[:2])
    print(len(repositoryOfQuestionsIndividual))
    return repositoryOfQuestionsGrouped, questionCountToQuestion, questionCountToQuestionCategory