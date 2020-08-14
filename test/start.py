from fastapi import FastAPI, UploadFile, File
import uvicorn
import json
from model import hw_classifier, make_df
#import LSTM_model

app=FastAPI()

@app.get("/")
async def root():
    return {"message":"hello world" }

@app.post("/predict/{index}")
async def create_file(index: str, file: UploadFile=File(...)):
    content= await file.read()
    json_obj = json.loads(content.decode('utf8'))
    if index=="LSTM":
       # model=LSTM_model.NET.load_model("checkpoint.pth")# тут файл с моделью
        return {"predict": "lstm is work"}
    elif index=="HW":
        pred=hw_classifier(json_obj)
        return {"predict": pred}
    else:
        return {"result": 0}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
