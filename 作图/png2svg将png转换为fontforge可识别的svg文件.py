#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import time
import operator
from collections import deque
from io import StringIO
from optparse import OptionParser

from PIL import Image


logging.basicConfig(level=logging.INFO)
log = logging.getLogger('png2svg')


def add_tuple(a, b):
    return tuple(map(operator.add, a, b))

def sub_tuple(a, b):
    return tuple(map(operator.sub, a, b))

def magnitude(a):
    return int(pow(pow(a[0], 2) + pow(a[1], 2), .5))

def normalize(a):
    mag = magnitude(a)
    assert mag > 0, "Cannot normalize a zero-length vector"
    return tuple(map(operator.truediv, a, [mag]*len(a)))

def svg_header(width, height):
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" preserveAspectRatio="xMidYMid meet" viewBox="0 0 %d %d">
""" % (width, height)

def joined_edges(assorted_edges, keep_every_point=False):
    pieces = []
    piece = []
    directions = deque([
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
    ])
    while assorted_edges:
        if not piece:
            piece.append(assorted_edges.pop())
        current_direction = normalize(sub_tuple(piece[-1][1], piece[-1][0]))
        while current_direction != directions[2]:
            directions.rotate()
        for i in range(1, 4):
            next_end = add_tuple(piece[-1][1], directions[i])
            next_edge = (piece[-1][1], next_end)
            if next_edge in assorted_edges:
                assorted_edges.remove(next_edge)
                if i == 2 and not keep_every_point:
                    piece[-1] = (piece[-1][0], next_edge[1])
                else:
                    piece.append(next_edge)
                if piece[0][0] == piece[-1][1]:
                    if not keep_every_point and normalize(sub_tuple(piece[0][1], piece[0][0])) == normalize(sub_tuple(piece[-1][1], piece[-1][0])):
                        piece[-1] = (piece[-1][0], piece.pop(0)[1])
                    pieces.append(piece)
                    piece = []
                break
        else:
            raise Exception("Failed to find connecting edge")
    return pieces

def rgba_image_to_svg_contiguous(im, opaque=True, keep_every_point=False):
    start_time = time.time()
    log.info(f"[{time.ctime()}] 开始处理图像：{im.size}")

    adjacent = ((1, 0), (0, 1), (-1, 0), (0, -1))
    visited = Image.new("1", im.size, 0)

    color_pixel_lists = {}

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            if visited.getpixel(here):
                continue
            rgba = im.getpixel((x, y))
            if opaque and not rgba[3]:
                continue
            piece = []
            queue = [here]
            visited.putpixel(here, 1)
            while queue:
                here = queue.pop()
                for offset in adjacent:
                    neighbour = add_tuple(here, offset)
                    if not (0 <= neighbour[0] < width) or not (0 <= neighbour[1] < height):
                        continue
                    if visited.getpixel(neighbour):
                        continue
                    neighbour_rgba = im.getpixel(neighbour)
                    if neighbour_rgba != rgba:
                        continue
                    queue.append(neighbour)
                    visited.putpixel(neighbour, 1)
                piece.append(here)

            if not rgba in color_pixel_lists:
                color_pixel_lists[rgba] = []
            color_pixel_lists[rgba].append(piece)

    log.info(f"[{time.ctime()}] 像素遍历完成，耗时: {time.time() - start_time:.2f} 秒")

    edges = {
        (-1, 0): ((0, 0), (0, 1)),
        (0, 1): ((0, 1), (1, 1)),
        (1, 0): ((1, 1), (1, 0)),
        (0, -1): ((1, 0), (0, 0)),
    }

    color_edge_lists = {}

    for rgba, pieces in color_pixel_lists.items():
        for piece_pixel_list in pieces:
            piece_set = set(piece_pixel_list)
            edge_set = set()
            for coord in piece_pixel_list:
                for offset, (start_offset, end_offset) in edges.items():
                    neighbour = add_tuple(coord, offset)
                    if neighbour not in piece_set:
                        start = add_tuple(coord, start_offset)
                        end = add_tuple(coord, end_offset)
                        edge = (start, end)
                        edge_set.add(edge)
            if not rgba in color_edge_lists:
                color_edge_lists[rgba] = []
            color_edge_lists[rgba].append(edge_set)

    log.info(f"[{time.ctime()}] 边界提取完成，耗时: {time.time() - start_time:.2f} 秒")

    color_joined_pieces = {}

    for color, pieces in color_edge_lists.items():
        color_joined_pieces[color] = []
        for assorted_edges in pieces:
            color_joined_pieces[color].append(joined_edges(assorted_edges, keep_every_point))

    log.info(f"[{time.ctime()}] 路径构建完成，耗时: {time.time() - start_time:.2f} 秒")

    s = StringIO()

    path_data = []

    for color, shapes in color_joined_pieces.items():
        r, g, b, a = color
        if a == 0:
            continue  # 跳过透明路径
        if (r, g, b) != (0, 0, 0):  # 只保留黑色
            continue
        for shape in shapes:
            for sub_shape in shape:
                here = sub_shape.pop(0)[0]
                path_data.append(f' M {here[0]} {here[1]} ')
                for edge in sub_shape:
                    here = edge[0]
                    path_data.append(f' L {here[0]} {here[1]} ')
                path_data.append(' Z ')

    if not path_data:
        s.write(' <path d="M 0 0 L 0 0" fill="none" stroke="none" />\n')
        s.write('</svg>\n')
        return s.getvalue()

    # 提取所有坐标点
    coords = []
    for line in path_data:
        if line.startswith(' M ') or line.startswith(' L '):
            parts = line.strip().split()
            x = int(parts[1])
            y = int(parts[2])
            coords.append((x, y))

    # 计算包围盒
    min_x = min(x for x, y in coords)
    max_x = max(x for x, y in coords)
    min_y = min(y for x, y in coords)
    max_y = max(y for x, y in coords)

    # 设置 viewBox 为 100x100
    viewBox_width = 100
    viewBox_height = 100

    # 计算缩放因子
    width_diff = max_x - min_x
    height_diff = max_y - min_y
    if width_diff == 0 or height_diff == 0:
        s.write(' <path d="M 0 0 L 0 0" fill="none" stroke="none" />\n')
        s.write('</svg>\n')
        return s.getvalue()

    scale_x = viewBox_width / width_diff
    scale_y = viewBox_height / height_diff
    scale = min(scale_x, scale_y)

    # 计算平移
    translate_x = -min_x * scale
    translate_y = -min_y * scale

    # 应用变换
    transformed_path_data = []
    for line in path_data:
        if line.startswith(' M ') or line.startswith(' L '):
            parts = line.strip().split()
            x = int(parts[1])
            y = int(parts[2])
            new_x = x * scale + translate_x
            new_y = y * scale + translate_y
            transformed_path_data.append(f' {parts[0]} {new_x:.2f} {new_y:.2f} ')
        else:
            transformed_path_data.append(line)

    # 生成 SVG
    s.write(svg_header(viewBox_width, viewBox_height))
    s.write(f' <path d="{" ".join(transformed_path_data)}" fill="black" stroke="none" />\n')
    s.write('</svg>\n')
    log.info(f"[{time.ctime()}] SVG 写入完成，总耗时: {time.time() - start_time:.2f} 秒")

    return s.getvalue()

def remove_gray_background_for_chinese_char(image_path, output_path, threshold=150):
    start = time.time()
    log.info(f"[{time.ctime()}] 开始去除灰色背景")
    img = Image.open(image_path).convert("RGBA")
    data = img.getdata()
    new_data = []
    for item in data:
        r, g, b, a = item
        gray = (r + g + b) // 3
        if gray > threshold:
            new_data.append((0, 0, 0, 0))  # 透明
        else:
            new_data.append((0, 0, 0, 255))  # 不透明黑色
    img.putdata(new_data)
    img.save(output_path, "PNG")
    log.info(f"[{time.ctime()}] 背景移除完成，耗时: {time.time() - start:.2f} 秒")


def png_to_svg(filename, contiguous=True, opaque=True, keep_every_point=False, remove_bg=True):
    start = time.time()
    log.info(f"[{time.ctime()}] 开始加载图像: {filename}")
    try:
        im = Image.open(filename)
    except IOError:
        sys.stderr.write('%s: Could not open as image file\n' % filename)
        sys.exit(1)

    log.info(f"[{time.ctime()}] 图像加载完成，耗时: {time.time() - start:.2f} 秒")

    if remove_bg:
        temp_path = filename.replace(".png", "_temp.png")
        remove_gray_background_for_chinese_char(filename, temp_path)
        im = Image.open(temp_path)
        im_rgba = im.convert('RGBA')
        os.remove(temp_path)
        log.info(f"[{time.ctime()}] 图像转换为 RGBA，耗时: {time.time() - start:.2f} 秒")
    else:
        im_rgba = im.convert('RGBA')

    result = rgba_image_to_svg_contiguous(im_rgba, opaque, keep_every_point)

    log.info(f"[{time.ctime()}] 总耗时: {time.time() - start:.2f} 秒")

    return result


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-v", "--verbose", action="store_true", dest="verbosity", help="显示详细调试信息", default=None)
    parser.add_option("-q", "--quiet", action="store_false", dest="verbosity", help="不显示警告信息", default=None)
    parser.add_option("-p", "--pixels", action="store_false", dest="contiguous", help="为每个像素生成一个形状，不合并相同颜色的区域", default=True)
    parser.add_option("-o", "--opaque", action="store_true", dest="opaque", help="仅处理不透明像素，忽略透明像素", default=True)
    parser.add_option("-1", "--one", action="store_true", dest="keep_every_point", help="保留所有点，即使在直线上", default=False)
    parser.add_option("--no-bg", action="store_false", dest="remove_bg", help="不移除灰色背景", default=True)
    parser.add_option("--output", dest="output", help="指定输出文件路径（可选）", default=None)

    (options, args) = parser.parse_args()

    if options.verbosity == True:
        log.setLevel(logging.DEBUG)
    elif options.verbosity == False:
        log.setLevel(logging.ERROR)

    if len(args) < 1:
        for file in os.listdir("."):
            if file.endswith(".png"):
                print(f"正在转换文件: {file}")
                output_file = options.output or file.replace(".png", ".svg")
                with open(output_file, 'w') as f:
                    f.write(png_to_svg(file, contiguous=options.contiguous, opaque=options.opaque, keep_every_point=options.keep_every_point, remove_bg=options.remove_bg))
                print(f"转换后的文件为：{output_file}")
    else:
        for file in args:
            if file.endswith(".png"):
                print(f"正在转换文件: {file}")
                output_file = options.output or file.replace(".png", ".svg")
                with open(output_file, 'w') as f:
                    f.write(png_to_svg(file, contiguous=options.contiguous, opaque=options.opaque, keep_every_point=options.keep_every_point, remove_bg=options.remove_bg))
                print(f"转换后的文件为：{output_file}")

# 参考代码：https://github.com/montesquio/png2svg/blob/master/png2svg.py
# 若原始图像不清晰，可以在去除背景色之前，使用Real-ESRGAN模型提升图像清晰度

