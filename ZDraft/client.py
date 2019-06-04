# coding: utf-8

import json
import time
import struct
import socket


def rpc(sock, in_, params):
    request = str.encode(json.dumps({"in": in_, "params": params}))
    length_prefix = struct.pack("I", len(request))
    sock.sendall(length_prefix)
    sock.sendall(request)

    length_prefix = sock.recv(4)    # blocking if receive buffer is empty
    length, = struct.unpack("I", length_prefix)
    
    body = sock.recv(length)
    response = json.loads(body)

    return response["out"], response["result"]

def receive(sock, n):
    rs = []
    while n > 0:
        r = sock.recv(n)
        if not r:             # EOF
            return rs
        rs.append(r)
        n -= len(r)
    return ''.join(rs)

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 8080))
    for i in range(10):
        out, result = rpc(s, "ping", "ireader %d" % i)
        print(out, result)
        time.sleep(1)         # sleep 1s for observation
    s.close()




