import argparse
import asyncio
import websockets

global args


@asyncio.coroutine
def hello():
    global args
    websocket = yield from websockets.connect(f"ws://localhost:{args.port}/")

    print("Connected!")

    # name = input("What's your name? ")
    # yield from websocket.send(name)
    # print("> {}".format(name))

    # greeting = yield from websocket.recv()
    # print("< {}".format(greeting))

    yield from websocket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(hello())
