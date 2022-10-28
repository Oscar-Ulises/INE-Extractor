from fastapi import FastAPI, File, UploadFile
import Clasificador
app = FastAPI()

@app.post('/api/')
async def predicted_imagen(file_0: UploadFile = File(...),file_1: UploadFile = File(...)):
    return Clasificador.INE_MASTER(file_0,file_1)
    
    
    
    
    
    
    
    
    



