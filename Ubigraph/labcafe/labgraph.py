#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from time import sleep

import pdb
import xmlrpclib
import getopt
import sys
import re

# Create an object to represent our server.
server_url = 'http://127.0.0.1:20738/RPC2'
server = xmlrpclib.Server(server_url)
G = server.ubigraph
config = {}

# Attribute 1 ?
colors = {
    '0' : '#82ae46',
    '1' : '#f6ad49',
    '2' : '#bce2e8',
    '3' : '#f2a0a1'
}

node_sizes = {}

def usage():
    print "labgraph.py [--from yymmdd] data.csv"

data_regex = re.compile(r"(\d\d)\.(\d\d)\.(\d\d)")
def split_date(s):
    m = data_regex.match(s)
    if not m:
        print("Invalid date format : %s" % s)
        exit(1)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

def next_month(d):
    """ d : (2, 3, 5) """
    (year, month, _) = d
    if month == 12:
        return (year + 1, 1, 1)
    return (year, month + 1, 1)

def sort_by_date(data):
    """
    Given data from read_data, construct a time-oriented sequence

    {
      "axis" : [(8,7,1) ... (9,7,1)]
      "data" : { (8,7,1) : [ { id, from, name, ... } ... ],
                 (8,8,1) : [ ... ],
                 (8,9,1) : [ ... ],
                 ...
               }
    }

    """
    ret = {}
    if len(data) == 0:
        print("data is zero")
        exit(1)
    min = data[0]['date']
    max = data[0]['date']
    for e in data:
        d = e['date']
        if d < min:
            min = d
        if d > max:
            max = d
    periods = []
    t = (min[0], min[1], 1)
    while True:
        if t > max:
            break
        periods.append(t)
        t = next_month(t)
    ret['axis'] = periods
    ret['data'] = {}
    x = ret['data']
    for e in data:
        d = e['date']
        k = (d[0], d[1], 1)
        if not k in x:
            x[k] = []
        x[k].append(e)
    return ret

def read_data(path):
    """
    reads the file and creates following data structure

    [ { id,
        from,
        name,
        date,
        affi,
        dept,
        attr1, attr2, attr3, attr4
      } ]
    """
    ret = []
    try:
        f = open(path, 'r')
    except IOError:
        print("Could not open data file: %s" % path)
        exit(1)
    for line in f:
        l = line.decode('utf-8')
        # line consists of
        # 0     1       2         3      4      5      6     7     8     9
        # id    name    date      from   affi   dept   attr1 attr2 attr3 attr4
        # 10,   tomo    08.07.15, 5,     東京学芸大学院, 教育,   1,    2,    2,    2,
        e = l.split(',')
        if e[0] == '':
            break
        new_element = {
            "id"    : int(e[0]),
            "name"  : e[1],
            "date"  : split_date(e[2]),
            "from"  : int(e[3]) if e[3] != '' else None,
            "affi"  : e[4],
            "dept"  : e[5],
            "attr1" : e[6],
            "attr2" : e[7],
            "attr3" : e[8],
            "attr4" : e[9]
            }
        ret.append(new_element)
    return ret

def add_node(n):
    """
    Puts a node into the graph
    """
    id = n['id']
    server.ubigraph.new_vertex_w_id(id)
    if n['from']:
        e = G.new_edge(n['from'], id)
        G.set_edge_attribute(e, 'arrow', 'true')
        G.set_edge_attribute(e, 'color', '#FFFFFF')
        G.set_edge_attribute(e, 'oriented', 'true')
        G.set_edge_attribute(e, 'strength', '0.4')
        k = node_sizes[n['from']] + 0.2
        node_sizes[n['from']] = k
        G.set_vertex_attribute(n['from'], 'size', str(k))

    else:
        print("%d does not have from" % id)
    G.set_vertex_attribute(id, 'color', colors[n['attr1']])
    G.set_vertex_attribute(id, 'label', str(id))
    if not id in node_sizes:
        node_sizes[id] = 1.0
    G.set_vertex_attribute(id, 'size', str(node_sizes[id]))
    G.set_vertex_attribute(id, 'shape', 'sphere')

#    G.set_vertex_attribute(id, 'width', '1.5')

    print((u"%s, %d年%d月" % (n['name'], 2000+n['date'][0], n['date'][1])).encode('utf-8'))

def show_graph_by_date(data):
    """
    Given data from sort_by_date, show the date periodically
    """
    server.ubigraph.clear()

    periods = data['axis']
    elements = data['data']

    for p in periods:
        if not p in elements:
            continue
        nodes = elements[p]
        for n in nodes:
            add_node(n)
        sleep(1)



def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:", ["help", "since="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-s", "--since"):
            config["since"] = a

    if len(args) != 1:
        print("Specify a data file")
        exit(1)

    data = read_data(args[0])
    sorted = sort_by_date(data)
    show_graph_by_date(sorted)
    return

if __name__ == '__main__':
    main()

