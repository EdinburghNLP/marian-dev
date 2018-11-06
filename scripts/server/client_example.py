#!/usr/bin/env python

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse

from websocket import create_connection


def add_empty_lines(text, positions, n_best, use_path_scores):
    n_best_mode = n_best >= 0
    output = []
    indices = set(positions)
    targetId = 0
    sentenceId = 0
    addedEmptyLines = 0
    for line in text.decode('utf-8').strip().split("\n"):
        if sentenceId in indices:
            if (not n_best_mode):
                output.append("")
            else:
                editedLine = "%s |||  ||| F0= 0.0 ||| 0.0" % sentenceId
                if use_path_scores:
                    editedLine += " ||| path_scores: 0.0"
                for k in range(n_best):
                    output.append(editedLine)
            addedEmptyLines += 1
            sentenceId = targetId + addedEmptyLines
            continue
        if (not n_best_mode):
            output.append(line)
            targetId += 1
        else:
            fields = line.split(" ||| ")
            targetId = int(fields[0])
            sentenceId = targetId + addedEmptyLines
            while sentenceId in indices:
              editedLine = "%s |||  ||| F0= 0.0 ||| 0.0" % sentenceId
              if use_path_scores:
                  editedLine += " ||| path_scores: 0.0"
              for k in range(n_best):
                  output.append(editedLine)
              addedEmptyLines += 1
              sentenceId = targetId + addedEmptyLines
            editedLine = " ||| ".join([str(targetId + addedEmptyLines)] + fields[1:])
            output.append(editedLine)
        sentenceId = targetId + addedEmptyLines
    targetId += 1
    sentenceId = targetId + addedEmptyLines
    while (addedEmptyLines < len(indices)):
        if (not n_best_mode):
            output.append("")
        else:
            editedLine = "%s |||  ||| F0= 0.0 ||| 0.0" % sentenceId
            if use_path_scores:
                editedLine += " ||| path_scores: 0.0"
            for k in range(n_best):
                output.append(editedLine)
        addedEmptyLines += 1
        sentenceId = targetId + addedEmptyLines

    return "\n".join(output)


if __name__ == "__main__":
    # handle command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("-n", "--n-best", type=int, default=0)
    parser.add_argument("-s", "--use-path-scores", default=False, action="store_true")
    args = parser.parse_args()

    # open connection
    ws = create_connection("ws://localhost:{}/translate".format(args.port))

    count = 0
    batch = ""
    empty_lines = []
    sentenceId = 0
    for line in sys.stdin:
        text = line.decode('utf-8') if sys.version_info < (3, 0) else line
        if not text.strip():
            empty_lines.append(sentenceId)
            sentenceId += 1
            continue
        count += 1
        sentenceId += 1
        batch += text
        if count == args.batch_size:
            # translate the batch
            ws.send(batch)
            result = ws.recv()
            if empty_lines:
                result = add_empty_lines(result, empty_lines, args.n_best, args.use_path_scores)
            print(result)

            count = 0
            batch = ""
            empty_lines = []

    if count:
        # translate the remaining sentences
        ws.send(batch)
        result = ws.recv()
        if empty_lines:
            result = add_empty_lines(result, empty_lines, args.n_best, args.use_path_scores)
        print(result)

    # close connection
    ws.close()
