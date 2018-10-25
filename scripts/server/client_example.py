#!/usr/bin/env python

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse

from websocket import create_connection


def add_empty_lines(text, positions):
    output = []
    indices = list(reversed(positions))
    for i, line in enumerate(text.rstrip().split("\n")):
        if indices:
            while indices and indices[-1] == i:
                output.append("")
                indices.pop()
        output.append(line)
    return "\n".join(output)


if __name__ == "__main__":
    # handle command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    # open connection
    ws = create_connection("ws://localhost:{}/translate".format(args.port))

    count = 0
    batch = ""
    empty_lines = []
    for line in sys.stdin:
        text = line.decode('utf-8') if sys.version_info < (3, 0) else line
        if not text.strip():
            empty_lines.append(count)
            continue
        count += 1
        batch += text
        if count == args.batch_size:
            # translate the batch
            ws.send(batch)
            result = ws.recv()
            if empty_lines:
                result = add_empty_lines(result, empty_lines)
            print(result.rstrip())

            count = 0
            batch = ""
            empty_lines = []

    if count:
        # translate the remaining sentences
        ws.send(batch)
        result = ws.recv()
        if empty_lines:
            result = add_empty_lines(result, empty_lines)
        print(result.rstrip())

    # close connection
    ws.close()
