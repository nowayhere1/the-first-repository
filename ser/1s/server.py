#!/usr/bin/env python

# WS server example

import asyncio
import websockets
import socket


async def hello(websocket, path):
    while True:
        print(path)
        name = await websocket.recv()
        print(f"< {name}")

        greeting = f"Hello {name}!"

        await websocket.send(greeting)
        print(f"> {greeting}")


if __name__ == "__main__":
    host = socket.gethostname()
    print(host)
    print(0)
    path  = 23
    start_server = websockets.serve(hello, host, 8010)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()