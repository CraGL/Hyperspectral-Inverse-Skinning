import asyncio
import websockets
import json

async def server( websocket, path ):
    async for msg in websocket:
        if msg.startswith( "bbw " ):
            parts = msg.split()
            nfaces, nvertices, laplacian_mode, solver_mode = parts[1:]
            nfaces = int( nfaces )
            nvertices = int( nvertices )
            
            faces    = await websocket.recv()
            vertices = await websocket.recv()
            handles  = await websocket.recv()
            
            faces    = np.frombuffer( faces,    dtype = np.int32   ).reshape( nfaces,    -1 ).copy()
            vertices = np.frombuffer( vertices, dtype = np.float32 ).reshape( nvertices, -1 ).copy()
            handles  = np.frombuffer( handles,  dtype = np.int32 ).copy()
            
            weights = bbw.bbw( faces, vertices, handles, laplacian_mode, solver_mode )
            weights = weights.astype(np.float32)
            
            await websocket.send( weights.tobytes() )
        
        elif msg.startswith( "linear_blend_skin_2D " ):
            parts = msg.split()
            nvertices, = parts[1:]
            nvertices = int( nvertices )
            
            vertices = await websocket.recv()
            weights  = await websocket.recv()
            handles  = await websocket.recv()
            
            vertices = np.frombuffer( vertices, dtype = np.float32 ).reshape( nvertices, -1 ).copy()
            handles  = np.array( json.loads( handles ) )
            weights  = np.frombuffer( weights,  dtype = np.float32 ).reshape( nvertices, len(handles) ).copy()
            
            deformed = bbw.linear_blend_skin_2D( vertices, weights, handles )
            
            await websocket.send( deformed.tobytes() )

start_server = websockets.serve( bbw_server, 'localhost', 9876)
## TODO: Call send and receive on the socket.

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
