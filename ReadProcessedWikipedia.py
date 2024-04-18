def getTopRanking(f1, topn):
    import pandas as pd
    print(f"Working with file {f1}")
    df = pd.read_csv(f1)

    # Sort the DataFrame based on 'col1' in descending order
    df_sorted = df.sort_values(by=["Length Of Response"], ascending=False)

    # Select the top 10 rows
    df_topN = df_sorted.head(topn)
    print(df_topN)
    topIDs = list(df_topN['WikiID'])
    print(f"Top IDs: {topIDs}")
    return topIDs

def getIDsToText(out_path_JSON, idOfInterest, columnOfInterest, saveToFilePath=None):
    import ijson
    if saveToFilePath != None:
        with open(saveToFilePath, "w", encoding='utf-8') as fwrite:
            with open(out_path_JSON, "rb") as f:
                for record in ijson.items(f, "item"):
                    if record["WikiID"] == idOfInterest:
                        fwrite.write(record['ID'])
                        fwrite.write("\n")
                        fwrite.write(record[columnOfInterest])
                        fwrite.write("\n")
        print(f'Written to {saveToFilePath}')
    else:
        with open(out_path_JSON, "rb") as f:
            for record in ijson.items(f, "item"):
                print(record)
                if record["WikiID"] == idOfInterest:
                    print(f"{record['ID']}: {record[columnOfInterest]}")

if __name__ == '__main__':

    folderOut = "/home/aleksei1985/Desktop/articles4/SQLiteDB/"

    #get file structure in place
    countToFolderName = {0: "1_RU2EN", 1: "2_EN", 2: "3_RU", 3: "4_EN2RU"}
    originalTextOptions = [folderOut + "RU2EN_Step2.json", folderOut + "EN_Step1.json", folderOut + "RU_Step1.json",
                   folderOut + "EN2RU_Step2.json"]
    filesUsedForRankOptions = []
    textToQAResponseAggregated = []
    for count in [0, 1, 2, 3]:
        out_path_JSON = folderOut + countToFolderName[count] + "Aggregate_Step4.json"
        textToQAResponseAggregated.append(out_path_JSON)

        outPathCSV = folderOut + countToFolderName[count] + "AggregateForRanking.csv"
        filesUsedForRankOptions.append(outPathCSV)

    #get top ranking WikiIDs for option
    optionToPlainText = {}
    optionToPlainText["1_RU2EN"] = "Russian Translated to English Articles"
    optionToPlainText["2_EN"] = "English Articles"
    optionToPlainText["3_RU"] = "Russian Articles"
    optionToPlainText["4_EN2RU"] = "English Translated to Russian Articles"

    folderOut = "UNCOVER/"
    import os
    if not os.path.exists(folderOut):
        os.makedirs(folderOut)
        print(f"The new directory {folderOut} is created!")

    #get top 100 WikiIDs
    topN = 100
    topIDsAcrossAllOptions = set([])
    for count in [0, 1, 2, 3]:
        print(f"Working with {optionToPlainText[countToFolderName[count]]}")
        print(f"Extracting Top {topN} WikiIDs (those with most negativity)")
        topIDs = getTopRanking(filesUsedForRankOptions[count], topN)
        for id in topIDs:
            topIDsAcrossAllOptions.add(id)

    print(len(topIDsAcrossAllOptions))
    print(topIDsAcrossAllOptions)

    for WikiID in ["Q49100"]:
        for count in [0, 1, 2, 3]:
            print(f"Working with {WikiID}")

            columnOfInterest = 'Text'
            outputPath = folderOut + WikiID + "_" + countToFolderName[count] + "_FileText.txt"
            #set outputPath to None to simply print to screen
            getIDsToText(originalTextOptions[count], WikiID, columnOfInterest, outputPath)

            columnOfInterest = 'Response'
            outputPath = folderOut + WikiID + "_" + countToFolderName[count] + "_GPTResponse.txt"
            #set outputPath to None to simply print to screen
            getIDsToText(textToQAResponseAggregated[count], WikiID, columnOfInterest, outputPath)
