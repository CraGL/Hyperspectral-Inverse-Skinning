'''
Our bpvf: 32*12*#bones/#vertices

This does not include the weight matrix, which is a fixed cost not dependent on the number of frames.

With the weight matrix, our bpvf is: 32*12*#bones/#vertices + (32*#vertices*#bones)/(#vertices*#frames)
'''

def bpvf_incremental( bones, vertices ):
    # 32 bits per float
    # 12 floats per handle matrix
    return 32*12*bones/float(vertices)

def bpvf_amortized( bones, vertices, frames ):
    # 32 bits per float
    return bpvf_incremental( bones, vertices ) + 32*bones/float(frames)

# name, vertices, poses, bones
data = [
    ( 'cat-poses', 7027, 9, 10 ),
    ( 'cat-poses', 7027, 9, 15 ),
    ( 'cat-poses', 7027, 9, 20 ),
    ( 'cat-poses', 7027, 9, 25 ),
    ( 'chickenCrossing', 3030, 400, 20 ),
    ( 'chickenCrossing', 3030, 400, 28 ),
    ( 'elephant-gallop', 42321, 48, 10 ),
    ( 'elephant-gallop', 42321, 48, 20 ),
    ( 'elephant-gallop', 42321, 48, 27 ),
    ( 'elephant-poses', 42321, 10, 10 ),
    ( 'elephant-poses', 42321, 10, 21 ),
    ( 'face-poses', 29299, 9, 27 ),
    ( 'horse-collapse', 8431, 53, 10 ),
    ( 'horse-collapse', 8431, 53, 20 ),
    ( 'horse-gallop', 8431, 48, 10 ),
    ( 'horse-gallop', 8431, 48, 20 ),
    ( 'horse-gallop', 8431, 48, 33 ),
    ( 'horse-poses', 8431, 10, 10 ),
    ( 'horse-poses', 8431, 10, 20 ),
    ( 'lion-poses', 5000, 9, 10 ),
    ( 'lion-poses', 5000, 9, 21 ),
    ( 'pcow', 2904, 204, 10 ),
    ( 'pcow', 2904, 204, 24 ),
    ( 'pdance', 7061, 201, 10 ),
    ( 'pdance', 7061, 201, 24 ),
    ( 'pjump', 15830, 222, 20 ),
    ( 'pjump', 15830, 222, 40 ),
        
    # kavan
    ( 'crane', 10002, 175, 40 ),
    ( 'elasticCow', 2904, 204, 18 ),
    ( 'elephant', 42321, 48, 25 ),
    ( 'horse', 8431, 48, 30 ),
    ( 'samba', 9971, 175, 30 )
    ]

print( "name,vertices,frames,bones,bpvf_incremental,bpvf_amortized" )

for datum in data:
    name, vertices, poses, bones = datum
    # print( datum + ( bpvf_incremental( bones, vertices ), bpvf_amortized( bones, vertices, poses ) ) )
    print( "%s,%s,%s,%s,%s,%s" % ( name, vertices, poses, bones, bpvf_incremental( bones, vertices ), bpvf_amortized( bones, vertices, poses ) ) )

# To filter, use `csvcut` as in:
## python for\ Guoliang\ bpvf.py | csvcut -C bones
