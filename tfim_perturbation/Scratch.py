#!/usr/bin/env python
# coding: utf-8

def constrained_partitions(n, k, min_elem, max_elem):
    allowed = range(max_elem, min_elem-1, -1)

    def helper(n, k, t):
        if k == 0:
            if n == 0:
                yield t
        elif k == 1:
            if n in allowed:
                yield t + (n,)
        elif min_elem * k <= n <= max_elem * k:
            for v in allowed:
                yield from helper(n - v, k - 1, t + (v,))

    return helper(n, k, ())

# Generate p_bar compositions
def p_bar(n):
    counter = 0
    for p in constrained_partitions(n,n+1,0,n):
        counter += 1
        print(p)
    print("Number of compositions: " + str(counter))

p_bar(3)

