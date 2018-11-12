#!/usr/bin/env python

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse

from websocket import create_connection

def correctIdOffset(result, batchIdOffset):
    output = []
    for line in result.strip().split("\n"):
        fields = line.split(" ||| ")
        internalId = int(fields[0])
        editedLine = " ||| ".join([str(internalId + batchIdOffset)] + fields[1:])
        output.append(editedLine)
    return "\n".join(output)

def translateBatch(batch, batchIdOffset, ws):
    ws.send(batch)
    result = ws.recv().decode('utf-8')
    result = correctIdOffset(result, batchIdOffset)
    return result.encode('utf-8')

def main():
    # handle command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size-tokens", type=int, default=24)
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("-n", "--n-best", type=int, default=0)
    parser.add_argument("-s", "--use-path-scores", default=False, action="store_true")
    args = parser.parse_args()

    # open connection
    ws = create_connection("ws://localhost:{}/translate".format(args.port))

    tokenCount = 0
    batch = ""
    sentenceId = 0
    batchIdOffset = 0
    for line in sys.stdin:
        text = line.decode('utf-8') if sys.version_info < (3, 0) else line
        sentenceLen = len(text.split()) + 1 	# also count the EoS
        if (tokenCount + sentenceLen > args.batch_size_tokens):
            # translate batch
            print(translateBatch(batch, batchIdOffset, ws))
            batchIdOffset = sentenceId
            tokenCount = 0
            batch = ""
        tokenCount += sentenceLen
        sentenceId += 1
        batch += text

    if (tokenCount > 0):
        # translate the remaining sentences
        print(translateBatch(batch, batchIdOffset, ws))

    # close connection
    ws.close()

if __name__ == "__main__":
    main()

