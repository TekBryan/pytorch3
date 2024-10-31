from ai_files.EmbeddingModel import BAIEmbeddingModel, OpenAIEmbeddingModel
import logging as log

from utils.RabbitMQ import publishMsgOnRabbitMQ



embedding_model = BAIEmbeddingModel()
OPENAI_embedding_model = OpenAIEmbeddingModel()

def check_All_Rows_Combined_Text_Length(rows, limit=7000):
    fileTexts = ''

    for row in rows:
        fileTexts += row

    if (len(fileTexts) > limit):
        return True
    
    return False

def embeddingTexts(arr, embedder="openai"):
    data = []

    if (embedder == "transformer"):
        # This line is using Transformer embedding function to embedd the text
        embedding_Texts = embedding_model.get_embedding_torch(arr)
    else:
        model = "text-embedding-3-small" if embedder.find("large") == -1 else "text-embedding-large"

        log.info(f"the model used for embedding is {model}")

        # This line is using OPENAI embedding function to embedd the text
        embedding_Texts = OPENAI_embedding_model.get_embedding(text=arr, retry=1, model=model)

    for idx, embedding_Text in enumerate(embedding_Texts):
        data.append({})

        for idx, embedding in enumerate(embedding_Text):
            data[-1]['emb_'+str(idx)] = embedding

    return data

async def embdeddingFunc(df, h, embedder="OpenAI", columns = [], selectedColumnIndex = 0, rowCount=1, totalRowCount=0, email=""):
    log.info("embedding start")
    data = []

    targetColumns = []
    rowTexts = []
    targetColumnName = ""
    headers = h
    for idx, row in df.iterrows():
        if (idx == 0):
            if (len(headers) < 1):
                headers = row.array
                targetColumnName = str(row.get(selectedColumnIndex))
            else:
                targetColumnName = headers[selectedColumnIndex]          

        fileText = ''

        for i, col in enumerate(columns):
            if (i != selectedColumnIndex):
                fileText += " " + str(headers[i]) + ": "+ str(row.get(col)) + " |"
            else:
                selectedColumnName = col

        # this is bug
        # fileText = targetColumnName + ": " + str(row.get(selectedColumnName)) + " |" + fileText

        if(idx == 1):
            log.info("file text value: ")
            log.info(fileText)

        targetColumns.append(str(row.get(selectedColumnName)))
        rowTexts.append(fileText)

        isLengthGreaterThenLimit = check_All_Rows_Combined_Text_Length(rowTexts, 7000)

        if (isLengthGreaterThenLimit or idx == df.shape[0] - 1):
            embeddedRows = embeddingTexts(rowTexts, embedder=embedder)

            for idx1, row in enumerate(embeddedRows):
                data.append({targetColumnName: targetColumns[idx1]})

                for col in row:
                    data[-1][col] = row[col]
            
            targetColumns = []
            rowTexts = []

        rowCount += 1
        embedding_percentage = rowCount / totalRowCount * 100

        # message for telling the embedding is done on a specific blob
        await publishMsgOnRabbitMQ({"task": "embedding", "condition": "continue", "percentage": str(embedding_percentage)}, email)


    return data, headers, rowCount

def embeddingFuncForInference(df, embedder="OpenAI", columns = []):
    data = []

    rowTexts = []
    labels = []
    for idx, row in df.iterrows():
        if (idx == 0):
            for r in row:
                labels.append(r)
            continue


        fileText = ''

        for i, col in enumerate(columns):
            fileText += " " + str(labels[i]) + ": "+ str(row.get(col)) + " |"
        
        fileText = fileText

        rowTexts.append(fileText)

        isLengthGreaterThenLimit = check_All_Rows_Combined_Text_Length(rowTexts, 7000)

        if (isLengthGreaterThenLimit or idx == df.shape[0] - 1):
            embeddedRows = embeddingTexts(rowTexts, embedder=embedder)

            for row in embeddedRows:
                data.append({})

                for col in row:
                    data[-1][col] = row[col]
            
            rowTexts = []

    return data