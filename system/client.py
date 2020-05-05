import asyncio
import websockets


@asyncio.coroutine
def hello():
    websocket = yield from websockets.connect("ws://localhost:8000/")

    name = input("What's your name? ")
    yield from websocket.send(name)
    print("> {}".format(name))

    greeting = yield from websocket.recv()
    print("< {}".format(greeting))

    yield from websocket.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(hello())
    # ws = websocket.WebSocket()
    # ws.connect("ws://localhost:8000")
