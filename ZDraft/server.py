# coding: utf-8

import json
import struct
import socket
import datetime

def handle_conn(conn, addr, handlers):
    print(addr, "comes")

    while True:
        
        length_prefix = conn.recv(4)
        if not length_prefix:
            print(addr, "bye")
            conn.close()
            break                   # exit while to handle next connection
        length, = struct.unpack("I", length_prefix)
        
        body = conn.recv(length)
        request = json.loads(body)

        in_ = request['in']
        params = request['params']
        print(in_, params)

        handler = handlers[in_]
        handler(conn, params)

def ping(conn, result):
    response = str.encode(json.dumps({"out": "pong", "result": result}))
    length_prefix = struct.pack("I", len(response))

    conn.sendall(length_prefix)
    conn.sendall(response)


def loop(sock, handlers):
    n = 0
    while True:
        n += 1
        print(datetime.datetime.now(), "Start serving [%d]" %n)
        conn, addr = sock.accept()
        handle_conn(conn, addr, handlers)


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # activate reuse addr 
    sock.bind(("localhost", 8080))
    sock.listen(1)

    # Register handlers
    handlers = {
        "ping": ping
    }

    # Service loop
    loop(sock, handlers)




