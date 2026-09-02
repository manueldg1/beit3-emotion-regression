from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()


class VA_Predictions(BaseModel):
    Valence: float
    Arousal: float
    Emotion: str | None = Noneclient_loop: send disconnect: Connection reset
PS C:\Users\Usuario\emotion_dataset>

class Interval_VA_Values(VA_Predictions):
    Valence_Interval: list(float)
    Arousal_Interval: list(float)



@app.post("/predict", )
async def create_user(user: UserIn) -> BaseUser:
    return user


# Blocco per permettere l'esecuzione con "python main.py"
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
