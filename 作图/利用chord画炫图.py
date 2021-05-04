#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install chord
# 注意，安装后，导入import Chord时，及html渲染 都需要连接公共网络才可以；
from chord import Chord

matrix = [
    [0, 5, 6, 4, 7, 4],
    [5, 0, 5, 4, 6, 5],
    [6, 5, 0, 4, 5, 5],
    [4, 4, 4, 0, 5, 5],
    [7, 6, 5, 5, 0, 4],
    [4, 5, 5, 5, 4, 0],
]
names = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Thriller"]

Chord(matrix, names).to_html(r'炫图.html')


def main():
    pass


if __name__ == '__main__':
    main()