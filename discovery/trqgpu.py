#!/usr/bin/python2.7

"""
Utilities to facilitate retrieval of available GPUs set by torque.
"""

import re
import os

import sys
import socket
import fcntl
import struct
import array

def all_interfaces():
    """
    Get all network interfaces.
    Notes
    -----
    http://code.activestate.com/recipes/439093-get-names-of-all-up-network-interfaces-linux-only/
    """

    is_64bits = sys.maxsize > 2**32
    struct_size = 40 if is_64bits else 32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    max_possible = 8 # initial value
    while True:
        bytes = max_possible * struct_size
        names = array.array('B', '\0' * bytes)
        outbytes = struct.unpack('iL', fcntl.ioctl(
            s.fileno(),
            0x8912,  # SIOCGIFCONF
            struct.pack('iL', bytes, names.buffer_info()[0])
        ))[0]
        if outbytes == bytes:
            max_possible *= 2
        else:
            break
    namestr = names.tostring()
    return [(namestr[i:i+16].split('\0', 1)[0],
             socket.inet_ntoa(namestr[i+20:i+24]))
            for i in range(0, outbytes, struct_size)]

def get_gpus():
    """
    Retrieve available GPUs set by torque.
    Returns
    -------
    result : dict
        Maps hostnames to lists of available GPU identifiers.
    """

    try:
        filename = os.getenv('PBS_GPUFILE')
    except:
        return {}
    else:
        if filename is None or not os.path.exists(filename):
            return {}
        else:
            result = {}
            with open(filename, 'r') as f:
                for line in f:
                    hostname, gpu = re.search('^(.*)-gpu(\d+)', line).groups()
                    if not result.has_key(hostname):
                        result[hostname] = []
                    result[hostname].append(int(gpu))
            return result

def cuda_visible_devices():
    """
    create CUDA_VISIBLE_DEVICES value corresponding based on available GPUs set by torque.
    Notes
    -----
    The returned value corresponds to that for the machine on which the function is run.
    """

    addr_list = [interface[1] for interface in all_interfaces() if interface[0] != 'lo']
    gpu_dict = get_gpus()
    for hostname in gpu_dict.keys():
        addr = socket.gethostbyname(hostname)
        if addr in addr_list:
            return ','.join(map(str, gpu_dict[hostname]))
    return ''

if __name__ == '__main__':
    result = cuda_visible_devices()
    if result:
        print result