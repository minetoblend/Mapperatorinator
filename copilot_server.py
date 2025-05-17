from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import AsyncIterator
import random 
from copilot import CopilotThread, ThreadStatus, CopilotRequest, ThreadState
import asyncio

thread = CopilotThread()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict]:
    print("starting copilot thread")

    thread.start()

    yield

    print("stopping copilot thread")

    thread.stop()
    

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/copilot")
async def copilot(request: CopilotRequest):
    if thread.state.status != ThreadStatus.IDLE:
        return { "success": False, "error": "thread_busy" }
    
    print("received request")

    uid = random.randint(0, 1_000_000_000)

    thread.state = ThreadState(status=ThreadStatus.REQUEST, uid=uid, request=request)

    while True:
        if thread.state.status == ThreadStatus.DONE and thread.state.uid == uid:
            result = thread.state.result

            assert result != None

            thread.state = ThreadState(status=ThreadStatus.IDLE)

            return { "success": True, "result": result }

        await asyncio.sleep(0.2)