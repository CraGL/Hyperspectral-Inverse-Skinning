#!/usr/bin/env python3

import websockets
import asyncio
import json
import os

port2pid = {}

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 9876

def send_data( data, port = None ):
    if port is None: port = DEFAULT_PORT
    port = int(port)
    
    setup_relay_server_if_needed( DEFAULT_HOST, port )
    
    ## Send a message to the relay server. Wait until it is sent to return.
    async def send_one():
        async with websockets.connect('ws://%s:%d' % (DEFAULT_HOST, port) ) as websocket:
            await websocket.send( json.dumps(data) )
    
    asyncio.get_event_loop().run_until_complete(send_one())


def port_is_in_use( host, port ):
    '''
    Returns True if the host:port is already in use.
    '''
    
    ## Adapted from: https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    import socket, errno
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        s.bind((host, port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            # print("Port is already in use")
            result = True
        else:
            # something else raised the socket.error exception
            print(e)
    
    s.close()
    return result

def setup_relay_server_if_needed( host, port, background = True ):
    ## Create a relay server if needed.
    if port_is_in_use( host, port ):
        print( "Host:port %s:%s already in use, hopefully by a relay server." % (host,port) )
        return
    
    if background:
        pid = os.fork()
        ## If we are not the forked process, return.
        if pid != 0: return
    
    ## If we are the forked process, setup a relay server.
    setup_relay_server( host, port )

def setup_relay_server( host, port ):
    port = int(port)
    
    connected = set()
    
    async def relay_server( websocket, path ):
        ## Remember this new websocket
        print( "New client:", websocket )
        connected.add( websocket )
        
        try:
            async for msg in websocket:
                ## Broadcast any message to all other clients.
                other_clients = set(connected)
                other_clients.remove( websocket )
                print( "Broadcasting message from", websocket, "to", len( other_clients ), "other clients." )
                if len( other_clients ) > 0:
                    await asyncio.wait([ client.send( msg ) for client in other_clients ])
        finally:
            print( "Client disconnected:", websocket )
            connected.remove( websocket )
    
    print( "Starting a relay server listening on %s:%s" % (host, port) )
    start_server = websockets.serve( relay_server, host, port )
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

def main():
    import argparse
    parser = argparse.ArgumentParser( description = "Echo messages to all connected WebSocket clients." )
    parser.add_argument( "--port", type = int, default = DEFAULT_PORT, help="The port to listen on." )
    args = parser.parse_args()
    
    setup_relay_server_if_needed( DEFAULT_HOST, args.port, background = False )

if __name__ == '__main__':
    main()

## I don't think the following is a good idea. There is a race condition:
## after this module is first loaded, but before the first call to send_data(),
## the HTML page must be loaded or refreshed (or attempt to reconnect to the websocket).
#else:
#    setup_relay_server_if_needed( DEFAULT_HOST, DEFAULT_PORT )
