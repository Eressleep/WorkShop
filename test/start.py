from fastapi import FastAPI, UploadFile, File
from enum import Enum
import uvicorn
import json
import HW_model
import LSTM_model

class ModelName(str, Enum):
    LSTM = "LSTM"
    HW = "HW"

app=FastAPI(title='Timeseries forecasting', description='Timeseries forecasting using LSTM and Holt-Winters model')




@app.get("/")
async def root():
    return {"message":"hello world" }

@app.post("/predict/{model_name}")
async def create_file(model_name: ModelName, file: UploadFile=File(...)):
    content= await file.read()
    json_obj = json.loads(content.decode('utf8'))
    if model_name = ModelName.LSTM:
        datX,datY=LSTM_model.create_dataset(json_obj, LSTM_model.LOOK_BACK)
        train = torch.randn(1,1, LSTM_model.LOOK_BACK)
        train[0][0] = datX[0]
        pred = LSTM_model.predict(model, train, 119) #пока не знаю что за загадочное число 119
        return {"predict": pred}

    if model_name=ModelName.HW:
        pred=HW_model.hw_classifier(json_obj)
        return {"predict": pred}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    model = Net(LSTM_model.input_dim, LSTM_model.hidden_dim, LSTM_model.output_dim)
    model.to(LSTM_model.device)
    model=LSTM_model.load_model('/resources/LSTM_model.pth')

