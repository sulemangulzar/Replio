from fastapi import FastAPI


app = FastAPI(
    title="Replio",
    description="An Email Automater",
    version="1.0.0",

)

@app.get("/health")
def get_health():
    return {"message" : "Healthy"}
