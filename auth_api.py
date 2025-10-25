from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext

app= FastAPI()
pwd_context= CryptContext(schemes=['bcrypt'], deprecated='auto')
users_db={}

class User(BaseModel):
    username: str
    password: str


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)


@app.post("/signup")
def signup(user:User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    users_db[user.username]= get_password_hash(user.password)
    return {'message': 'User created successfully'}

@app.post("/login")
def login(user: User):
    if user.username not in users_db or not verify_password(user.password, users_db[user.username]):

        raise HTTPException(status_code=400, detail="Invalid username or password")
    return {'message': 'Login successful'} 

