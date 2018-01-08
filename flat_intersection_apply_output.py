from __future__ import print_function, division

from numpy import *
import format_loader

def main():
    import argparse
    
    parser = argparse.ArgumentParser( description='Apply the transformations from the DMAT output of flat_intersection.py to the original mesh.' )
    parser.add_argument( 'rest_pose_path', type=str, help='Rest pose (OBJ).')
    parser.add_argument( 'DMAT_path', type=str, help='Transformations (DMAT).')
    parser.add_argument( 'output_path', type=str, help='Output path (OBJ).')
    
    args = parser.parse_args()
    
    Ts = format_loader.load_DMAT( args.DMAT_path ).T
    
    with open( args.rest_pose_path ) as input_OBJ:
        with open( args.output_path, 'w' ) as output_OBJ:
            
            count = 0
            for line in input_OBJ:
                if not line.startswith( 'v ' ):
                    output_OBJ.write( line )
                    continue
                
                x,y,z = [ float(v) for v in line.split()[1:] ]
                
                T = Ts[count].reshape(4,3).T
                Txyz = dot( T, [ x, y, z, 1 ] )
                
                # print( 'v', repr(xp), repr(yp), repr(zp), file = output_OBJ )
                output_OBJ.write( ' '.join( ['v'] + [ repr(v) for v in Txyz ] ) + '\n' )
                
                count += 1
    
    print( "Saved:", args.output_path )

if __name__ == '__main__':
    main()
