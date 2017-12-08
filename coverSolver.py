import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.3g" % x})
from itertools import product, permutations
import math
import matplotlib as mpl
import exact_cover_np


class Universe(object):
    ''' Defines the shape which is enough to retain geometric state of any shape.  
        The universe shape is used to convert a shape to a binary state vector.
        
        Also defines the coordinate basis in which the shapes are specified, 
        and color for underlying (unshifted, unrotated) pieces.'''
    def __init__(self,board, basis, pieceCount):
        self.board=board
        self.boardGeomState=geomState(board.ar)
        self.boardStateMap={s:i for i,s in enumerate(self.boardGeomState)}

        self.stateDims=len(self.boardGeomState)+pieceCount
        self.basis=basis #useful for 4-way symmetric lattices.
        self.invBasis=np.linalg.inv(basis) #useful for 4-way symmetric lattices.
        #a color for each piece index, to match a real-life puzzle piece color
        self.colors=(
            'xkcd:pink',
            'xkcd:darkblue',
            'lime',
            'xkcd:crimson',
            'xkcd:yellow',
            'silver',
            'xkcd:red',
            'xkcd:lightblue',
            'xkcd:darkgreen',
            'xkcd:purple',
            'xkcd:orange',
            'xkcd:beige',
        )

def geomState(ar):
    ''' @param ar: an array NxM coordinates in M-dimentional space
        @return a set of strings, where the strings represents one of N coordinates in 3D space.
    '''
    s=set()
    for c in ar:
        s.add(c.tobytes())
    return s

class Shape(object):
    '''A covering set, a.k.a a puzzle piece rotated and translated to cover a specific region of the universe.
        The intention is for it to be immutable: don't modify any member variables.
    '''
    
    def __init__(self,ar,idx):
        self.ar=ar
        self.idx=idx
        self.bbox=self._boundingBox()
        
    def __repr__(self):
        return 'Shape'+str((self.ar, self.idx))
    
    def _boundingBox(self):
        return np.array([
            np.min(self.ar,axis=0),
            np.max(self.ar,axis=0)
        ])
    
#     def grow(self, by):
#         '''return a piece like self, but grown by 'by' points in either direction along each of u.basis vectors in the lattice.
#             FIXME not implemented for trangular or hexagonal lattices.
#         '''
#         kern=list((product([-1,0,1],repeat=3))) #27 vectors for all the neighbors of a point in 3d grid, including edge and corner neighbors
#         kern=np.array(kern,dtype=np.int8)
#         grown=(kern+self.ar[:,None,:]) #pairwise add every tuple in kern to every tuple in ar.  See numpy broadcasting for why this works.
#         grown=np.vstack(grown)
#         return Shape(np.unique(grown,axis=0),self.idx)
    
    def orientations(self, rots):
        '''
            @param rots: a Nx3x3 array specifying N rotations, in the identity basis (not the u.basis).
            @return a Shapeset containing this shape rotated by all possible rots. the rotated shapes are shifted to have the 
            most bottom,left,into-the-screen corner of the bounding box at the origin.  Duplicates are removed.
        '''
        ors=np.zeros((len(rots),)+self.ar.shape,dtype=self.ar.dtype)
        for i,r in enumerate(rots):
            p=self.ar.dot(u.basis.T).dot(r.T).dot(u.invBasis.T)
            p=p-np.min(p,axis=0) #shift bounding box to origin
            pint=np.rint(p,dtype=r.dtype)
            if not np.allclose(p,pint):
                raise Exception('%s is not close to integers after %dth rotation by\n%s\nThe result is %s'%(self,i,r,p))
            ors[i]=pint
        ors = np.unique(ors,axis=0)
        shapes=[Shape(a,self.idx) for a in ors]
        return ShapeSet(shapes)
    
    def containedTranslations(self, board):
        ''' return a ShapeSet of all translations of self that is contained in other'''
        rng=board.bbox-self.bbox #first row is the start shift, second row is the end shift
        dimShifts=[range(rng[0][dim],rng[1][dim]+1) for dim in range(self.ar.shape[1])]
        shifts=np.array(list(product(*dimShifts)),dtype=self.ar.dtype)
#         print dimShifts
        validShapes=ShapeSet()
        for s in shifts:
            translatedAr=self.ar+np.array(s,dtype=self.ar.dtype)
            tps=geomState(translatedAr)
            if     tps <= u.boardGeomState:
                validShapes.append(Shape(translatedAr,self.idx))
        return validShapes
    
    def asStateVector(self):
        '''
            @return: a binary vector, with a one indicating the shape occupies a 
                slot on the board, augmented with 1 in the self.idx position
        '''
        if self.idx<0:
            raise ValueError('cannot create state vector for negative index'%self.idx)
        s=np.zeros(u.stateDims,dtype=np.int8)
        posState=[u.boardStateMap[v] for v in geomState(self.ar)]
        s[np.array(posState)]=True
        s[len(u.boardStateMap)+self.idx]=True
        return s
        
class ShapeSet(list):
    '''A collection of Shapes.'''
    def visualize(self, individually=False):
        '''draw a set of shapes in 3d'''
        allShapes=np.vstack([s.ar for s in self])
        allShapes
        X,Y,Z=np.hsplit(allShapes,3)
        #bounding box
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        for i,p in enumerate(self):
            if individually or i == 0:            
                ax=self._prepareAxis(Xb, Yb, Zb)
            color=u.colors[p.idx]
            size=500
#             elif p.idx==-1:#the board
#                 color='green'
#                 size=30
#             elif p.idx==-2:#the universe
#                 color='black'
#                 size=5
            inOrthogonal=p.ar.dot(u.basis.T)
            ax.scatter(*np.hsplit(inOrthogonal,3), c=np.array([mpl.colors.to_rgba(color,.7)]), marker='o', s=size, depthshade=False)
            plt.show()

    def _prepareAxis(self, Xb, Yb, Zb):
        fig = plt.figure()
        fig.set_size_inches(7,7)
        # Comment or uncomment following both lines to test the fake bounding box:
        ax = fig.add_subplot(111, projection='3d')
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        #plot the board
        inOrthogonal=u.board.ar.dot(u.basis.T)
        ax.scatter(*np.hsplit(inOrthogonal,3), c='green', marker='o', s=5)        
        ax.set_aspect('equal')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
            
        return ax
    
    def asStateMatrix(self):
        m=np.array([s.asStateVector() for s in self])
        return m
    
def rotations(axes, folds):
    ''' Generate rotation matrix, by picking a permutation from axes, rotating about each axis in the permutation by  360/fold degrees, 0...fold times.
        @param axes: Nx3 matrix, with each row an axis.
        @param folds: Nx1 vector of integers, corresponding to the axes, so that rotations are done in 360/fold degree increments
        @return returns a Nx3x3 array o, where each o[i,:,:] is a rotation matrix.  The matrices will be unique
    '''
    foldLists=[]#a list of lists, with one list per axis, with inner list containing a rot matrix for every fold
    for a,fold in zip(axes,folds):
        g=axisToRotationMatrix(a[None,:].T,2*math.pi/fold)
        foldLists.append([g**p for p in range(fold)])
    
    rots=[]
    for perm in permutations(foldLists):
        for axisRots in product(*perm):
            curRot=np.mat(np.eye(3))
            for r in axisRots:
                curRot *=r
            rots.append(np.around(curRot,10))
    #print ("generated %d rotations"%len(rots))
    return np.unique(np.stack(rots),axis=0)

def axisToRotationMatrix(axis, angleRads):
    '''return a rotation matrix to perform a rotation about axis by angleRads.
       This uses the right-hand rule coordinate system, X pointing right, Y up and Z pointing out of the screen.'''
    #could have also used quaternions here

    #create a matrix with axis as one of it's basis vecors, and the other two being orthonormal to it
    q,_=np.linalg.qr(axis,'complete')
    q=np.mat(q) #the 1 column is our axis arg

    #the rotation matrix around the x axis
    co,si=math.cos(angleRads),math.sin(angleRads)
    rot=np.matrix(
        [[1,   0,  0],
         [0, co, -si],
         [0, si, co]])
    
    #1) change basis to rename the first column as the X axis, 2) rotate around it, 3) rename the X axis back to be the axis arg
    return np.mat(q*rot*q.I)

def getPieces():
    '''define the pieces of our puzzle.  They are all planar.'''
    pieces=[
        #nightstick
        [
            [0,0],
            [1,0],
            [2,0],
            [3,0],
            [1,1]
        ],
        [
        #long uneven corner    
            [0,0],
            [1,0],
            [2,0],
            [3,0],
            [0,1]
        ],
        [
        #square
            [0,0],
            [1,0],
            [0,1],
            [1,1],
        ],
        [
        #letter p
            [0,0],
            [0,1],
            [0,2],
            [1,1],
            [1,2],
        ],
        [
        #bridge
            [0,0],
            [1,0],
            [2,0],
            [0,1],
            [2,1]
        ],
        [
        #cross
            [1,0],
            [1,1],
            [1,2],
            [0,1],
            [2,1],
        ],
        [
        #stairs
            [0,0],
            [0,1],
            [1,1],
            [1,2],
            [2,2],
        ],
        [
        #large corner
            [0,0],
            [1,0],
            [2,0],
            [0,1],
            [0,2],
        ],
        [
        #lightning
            [0,0],
            [0,1],
            [0,2],
            [1,2],
            [1,3],
        ],
        [
        #stick
            [0,0],
            [1,0],
            [2,0],
            [3,0],
        ],
        [
        #short uneven corner
            [0,0],
            [1,0],
            [2,0],
            [0,1]
        ],
        [
        #small corner
            [0,0],
            [0,1],
            [1,0],
        ]
    ]
    for i,p in enumerate(pieces):
        p=np.array(p,dtype=np.int8)
        withZaxis=np.append(p,np.zeros((p.shape[0],1),dtype=np.int8),axis=1)
        pieces[i]=Shape(withZaxis, i)
    return pieces

def pyramidBoard():
    #our board is a pyramid with a 5x5 base, but in our basis, 
    #one ridge of the pyramid is along the z-axis
    side=5
    board=np.zeros((side,side,side),dtype=np.int8)
    for z in range(side): 
        board[z,:side-z,:side-z]=1
    board=np.transpose(np.flip(np.nonzero(board),0).astype(np.int8))    
    # board=np.array([[0,0,0],
    #                 [0,0,1]
    #                ])
    board=Shape(board,-1)
    return board

def rectangleBoard():
    board=np.ones((11,5,1))
    board=np.transpose(np.array(np.nonzero(board))).astype(np.int8)    
    return Shape(board,-1)

def rectangleProblem():
    pieces=getPieces()
    board=rectangleBoard()
    basis=np.eye(3)
    #many Shape and ShapeSet methods (those making use of a Shape's state representation) require the universe u global variable defined
    global u
    u=Universe(board,basis,len(pieces))

    #the Z axis with 4-fold symmetry
    #and the and the Y axis with 2-fold symmetry (flips of the pieces)
    #each row is a rotation axis
    rots=rotations(np.array(
        [[0,0,1],
         [0,1,0]
        ]), 
        np.array([4,2]).T)

    return rots, pieces

def pyramidProblem():
    pieces=getPieces()
    board=pyramidBoard()

    basis=np.array([[1,0,0.5],
                    [0,1,0.5],
                    [0,0,2.0**.5/2]])

    #many Shape and ShapeSet methods (those making use of a Shape's state representation) require the universe u global variable defined
    global u
    u=Universe(board,basis,len(pieces))

    #diagonals in the X-Y plane, are the rotation axes, with 4-fold symmetry each
    #and the and the Y axis with 2-fold symmetry
    #each row is a rotation axis
    rots=rotations(np.array(
        [[1,1,0],
         [1,-1,0],
         [0,1,0]
        ]), 
        np.array([4,4,2]).T)

    return rots, pieces

def solve(rots, pieces):    
    stateSet=ShapeSet()
    for p in pieces:
        for o in p.orientations(rots):
            ts=o.containedTranslations(u.board)
            stateSet.extend(ts)

    S = stateSet.asStateMatrix().astype(np.int32)
    print "number of oriented, positioned shapes (cover candidates): %d"%len(S) 
    solState = exact_cover_np.get_exact_cover(S)
    solSet=ShapeSet()
    for i in solState:
        solSet.append(stateSet[i])
        
    return solSet
    
if __name__ == '__main__':
#     p=pyramidProblem()
    p=rectangleProblem()
    s= solve(*p)
    print s