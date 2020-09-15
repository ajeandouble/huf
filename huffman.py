import sys
import logging
import random
import copy
from random import randint
from datetime import datetime
import time #dbg
from textwrap import wrap
import os
import math

logger = logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.disable()


class Node:
    def __init__(self, content, freq, lchild=None, rchild=None):
        self.content = content
        self.freq = freq
        self.lchild = lchild
        self.rchild = rchild

    def __lt__(self, other):
        return self.freq < other.freq
    def __gt__(self, other):
        return self.freq > other.freq
    def __le__(self, other):
        return self.freq <= other.freq
    def __ge__(self, other):
        return self.freq >= other.freq
    def __eq__(self, other):
        return self.freq == other.freq
    def __int__(self):
        return self.freq
    def __add__(self, other):
        return self.freq + other.freq
    def __sub__(self, other):
        return self.freq - other.freq
    def __repr__(self):
        return f'{self.content}:{self.freq}'

class MinHeap:
    def __init__(self):
        self.heap = []

    def __swap_items(self, i, j):
        tmp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = tmp

    def __bubble_up(self, i):
        while i > 0:
            if self.heap[i] < self.heap[(i - 1) // 2]:
                self.__swap_items(i, (i - 1) // 2)
                i = (i - 1) // 2
            else:
                break

    def insert(self, item):
        self.heap.append(item)
        self.__bubble_up(len(self.heap) - 1)
        
    def __bubble_down(self, i):
        last_i = len(self.heap) - 1 # Last element index
        while last_i >= 2 * i + 1:
            if last_i == 2 * i + 1:
                if self.heap[i] > self.heap[last_i]:
                    self.__swap_items(i, last_i)
                    #logging.debug(self.heap)
                break
            lchild = self.heap[i * 2 + 1]
            rchild = self.heap[i * 2 + 2]
            if self.heap[i] < lchild and self.heap[i] < rchild:
                break
            if lchild < rchild:
                self.__swap_items(i, i * 2 + 1)
                i = i * 2 + 1
            else:
                self.__swap_items(i, i * 2 + 2)
                i = i * 2 + 2
            #logging.debug(self.heap)

    def del_min(self):
        if not self.heap:
            return None

        deleted_root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        if len(self.heap) > 1:
            self.__bubble_down(0)
            pass

        return deleted_root
    
def isMinHeap(lst):
	try:
		for k in range(0, len(lst) // 2):
			if lst[k] > lst[k*2+1] or lst[k] > lst[k*2+2]:
				logging.error(f'KO: {lst[k]} > {lst[k*2+1]} or {lst[k*2+2]}')
				raise 
				return
	except IndexError:
		pass
	
	logging.debug('\tO.K.')

class HuffCoding:
    def __init__(self, min_heap):
        self.tree = None
        self.codes = {}
        self.binary_map = {}

        self.build_tree(min_heap)

    def test_huff_tree(self, node, level):
        if node:
            self.test_huff_tree(node.lchild, level + 1)
            padding = "-" * level * 8 + ">"
            logging.debug(f'{padding}{node.freq}:{node.content}')
            self.test_huff_tree(node.rchild, level + 1)

    def build_tree(self, min_heap):
        H = copy.deepcopy(min_heap)
        while len(H.heap) > 1:
            node_l = H.del_min()
            node_r = H.del_min()
            merged = Node(None, node_l.freq + node_r.freq, node_l, node_r)
            H.insert(merged)

        self.tree = H
        self.test_huff_tree(H.heap[0], 0)

    def make_codes(self):
        def make_codes_recursion(node, current_code):
            if not node:
                return

            if node.content is not None:
                self.codes[node.content] = current_code
                self.binary_map[current_code] = node.content
            make_codes_recursion(node.lchild, current_code + "0")
            make_codes_recursion(node.rchild, current_code + "1")
        make_codes_recursion(self.tree.heap[0], '')
        logging.debug(f'codes={self.codes}')
        return (self.codes, self.binary_map)
            
def make_frequency_dict(data):
    frequency = {}

    logging.debug(type(data))
    for content in data:
        if not content in frequency:
            frequency[content] = 0
        frequency[content] += 1
        #print(content)
    return frequency


def get_encoded_binary(data, codes):
    encoded_str = ""
    for byte in data:
        encoded_str += codes[byte]
    logging.info(f'len(encoded_str)={len(encoded_str)}, encoded_str={encoded_str}')

    if len(encoded_str) % 8 == 0:
        padding = 0
    else:
        padding = 8 - len(encoded_str) % 8

    logging.info(f'padding={padding}')
    encoded_binary = bytearray()
    for i in range(0, len(encoded_str), 8):
        byte = encoded_str[i:i+8]
        encoded_binary.append(int(byte, 2))

    logging.info(f'encoded_binary={encoded_binary}')
    return encoded_binary, padding

def create_header(filename, codes, padding, binary_map, encoded_binary):
    '''                
    [filename length][filename][number of codes][padding][codes]
    '''
    table = bytearray()
    filename = bytes(filename, 'utf-8')
    table += len(filename).to_bytes(1, 'little')
    table += filename
    table += len(codes).to_bytes(2, 'little') # Number of codes
    
    binary_codes = ''
    codes_table = bytearray()
    for code, binary in codes.items():
        codes_table += code.to_bytes(1, 'little')
        codes_table += len(binary).to_bytes(1, 'little')
        binary_codes += binary
    binary_codes = wrap(binary_codes, 8)
    
    logging.debug(f'binary_codes={binary_codes}')
    table += padding.to_bytes(1, 'little')
    table += codes_table
    binary_codes[-1] = binary_codes[-1].ljust(8, '0')
    logging.debug(f'binary_codes={binary_codes}')
    encoded_binary_codes = bytearray()
    for byte in binary_codes:
        encoded_binary_codes  += int(byte, 2).to_bytes(1, 'little')
    
    logging.info(f'len(table)={len(table)}\tlen(codes_table)={len(codes_table)}')
    logging.info(f'len(encoded_binary_codes={len(encoded_binary_codes)}')

    return table + encoded_binary_codes

def compress(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = bytearray()
            byte = file.read(1)
            while byte != b"":
                data += byte
                byte = file.read(1)
                #logging.debug(f'byte={byte}')
            
            frequency = make_frequency_dict(data)
            logging.debug(f'len(frequency)={len(frequency)}')
            logging.debug(frequency)
            H = MinHeap()
            for content, freq in frequency.items():
                H.insert(Node(content, freq))
            isMinHeap(H.heap)
            huffman = HuffCoding(H)
            codes, binary_map = huffman.make_codes()
            logging.debug(codes)
            logging.debug(f'len of data:{len(data)}')
            encoded_binary, padding = get_encoded_binary(data, codes)
            filename = os.path.split(path)[1]
            header = create_header(filename, codes, padding, binary_map, encoded_binary)
            data = header + encoded_binary

        output_filename, _ = os.path.splitext(filename)
        output_filename += '.huf'
        breakpoint()
        with open(output_filename, 'wb') as compressed_file:
            compressed_file.write(data)
    else:
        print(f'File {path} doesnt exist')

def decompress(path):
    with open(path, 'rb') as file:
        data = bytearray()
        byte = file.read(1)
        while byte != b"":
            data += byte
            byte = file.read(1)
    filename, binary_map, padding, data = parse_file(data)
    deflated_data = deflate_data(binary_map, padding, data)

    with open(filename, 'wb') as output_file:
        output_file.write(deflated_data)
        print(f'File {filename} successfully deflated')

def parse_file(data):
    filename_len = int(data[0])
    filename = data[1:filename_len + 1].decode('utf-8')
    codes_len = data[filename_len + 1: filename_len + 3]
    codes_len = int.from_bytes(data[filename_len + 1: filename_len + 3], byteorder='little', signed=False)
    padding = data[filename_len + 3]
    codes_table = data[filename_len + 4:filename_len + 4 + codes_len * 2]

    binary_codes_len = 0
    i = 0
    while i < codes_len * 2:
        binary_codes_len += int(data[filename_len + 4 + i + 1])
        i += 2

    logging.debug(f'binary_codes_len={binary_codes_len}')
    # ERREUR DANS LA TABLE DES CODES
    logging.debug(f'bcodelen={binary_codes_len}')
    start = 4 + filename_len + codes_len * 2
    end =  math.ceil(binary_codes_len / 8) + start
    binary_codes_bin = data[start:end]

    binary_codes_str = ""
    for byte in binary_codes_bin:
        binary_codes_str += bin(byte)[2:].rjust(8, '0')
        #print(bin(byte)[2:].rjust(8, '0'), '<-bc')
    #binary_codes_str = binary_codes_str[:binary_codes_len]

    #logging.debug(binary_codes_str, len(binary_codes_str))
    i = 0
    j = 0
    binary_map = {}
    while i < len(codes_table):
        c = int(codes_table[i])
        k = int(codes_table[i+1])
        binary = binary_codes_str[j:j+k]
        binary_map[binary_codes_str[j:j+k]] = c
        j += k
        i += 2

    # OFFSET
    logging.debug(f'parse_file padding={padding}')
    breakpoint()
    
    return filename, binary_map, padding, data[end:]

def deflate_data(binary_map, padding, data):
    bytes_str = ""
    i = 0
    while i < len(data) - 1:
        byte = bin(data[i])[2:]
        logging.debug(f'byte={byte}')
        byte = byte.rjust(8, '0')
        bytes_str += byte
        i += 1
    
    last_byte = ''
    try:
        last_byte = bin(data[i])[2:].rjust(8, '0')[padding:]
    except:
        pass
    bytes_str += last_byte

    logging.debug(f'len(bytes_str)={len(bytes_str)}\tbytes_str={bytes_str}')
    
    binary = ""
    decoded_text = ""
    decoded_file = bytearray()  
    for c in bytes_str:
        binary += c
        if binary in binary_map:
            decoded_text += chr(binary_map[binary])
            logging.debug(f'decoded_byte={hex(binary_map[binary])}')
            decoded_file += binary_map[binary].to_bytes(1, 'little')
            binary = ""
    
    logging.debug(f'decoded_file={decoded_file}')
    return decoded_file

def main():
    if len(sys.argv) != 3 or sys.argv[1] != "-c" and sys.argv[1] != "-d":
        print("Usage: %s [-cd] input" % sys.argv[0])
        exit (1)
    if (sys.argv[1] == '-c'):
        compress(sys.argv[2])
    if (sys.argv[1] == '-d'):
        decompress(sys.argv[2])

if __name__== '__main__':
    main()
