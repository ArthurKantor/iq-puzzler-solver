import matplotlib.pyplot as plt
import numpy as np
from dlmatrix import DLMatrix
from itertools import product, permutations
import math
import matplotlib as mpl
from alg_x import AlgorithmX

np.set_printoptions(formatter={'float_kind': lambda x: "%.3g" % x})

ShapeSet = list['Shape']


class Shape:
    """A covering set, a.k.a. a puzzle piece rotated and translated to cover a specific region of the universe.
        The intention is for it to be immutable: don't modify any member variables.
    """

    def __init__(self, ar, idx):
        self.ar: np.ndarray = ar
        self.idx: int = idx
        self.bbox = self._bounding_box()
        self.geom_state = self._compute_geom_state()

    def __repr__(self):
        return 'Shape' + str((self.ar, self.idx))

    def _bounding_box(self) -> np.ndarray:
        return np.array([
            np.min(self.ar, axis=0),
            np.max(self.ar, axis=0)
        ])

    def _compute_geom_state(self) -> frozenset[bytes]:
        """
            @return a set of strings, where each string represents a ball of the shape
             ( a coordinates in the grid space).
        """
        return frozenset(c.tobytes() for c in self.ar)

    def orientations(self, rots, basis) -> ShapeSet:
        """
            @param rots: a Nx3x3 array specifying N rotations, in the identity basis (aka in the real-world space).
            @param basis: a 3x3 array, which is the basis taking coordinates from grid space into the real-world space.
            @return a ShapeSet containing this shape (in grid space) rotated by all possible rots. the rotated shapes
            are shifted to have the most bottom,left,into-the-screen corner of the bounding
            box at the origin.  Duplicates are removed.
        """
        inv_basis = np.linalg.inv(basis)
        ors = np.zeros((len(rots),) + self.ar.shape, dtype=self.ar.dtype)
        for i, r in enumerate(rots):
            p = self.ar.dot(basis.T).dot(r.T).dot(inv_basis.T)
            p = p - np.min(p, axis=0)  # shift bounding box to origin
            pint = np.rint(p, dtype=r.dtype)
            if not np.allclose(p, pint):
                raise Exception(
                    '%s is not close to integers after %dth rotation by\n%s\nThe result is %s' % (self, i, r, p))
            ors[i] = pint
        # sort the balls within each shape by the (x,y,z) key
        sorted_indexes = np.lexsort((ors[:, :, 2], ors[:, :, 1], ors[:, :, 0]))
        # [[0 0 ...], [1 1 ...], ..., [23 23 ...]]  I miss matlab's repmat
        tiled_range = np.tile(np.arange(sorted_indexes.shape[0])[:, None], (1, sorted_indexes.shape[1]))
        ors = np.unique(ors[tiled_range, sorted_indexes], axis=0)
        shapes = [Shape(a, self.idx) for a in ors]
        return ShapeSet(shapes)

    def contained_translations(self, board) -> ShapeSet:
        """ return a ShapeSet of all translations of self that is contained in other"""
        rng = board.bbox - self.bbox  # first row is the start shift, second row is the end shift
        dim_shifts = [range(rng[0][dim], rng[1][dim] + 1) for dim in range(self.ar.shape[1])]
        shifts = np.array(list(product(*dim_shifts)), dtype=self.ar.dtype)
        valid_shapes = ShapeSet()
        for s in shifts:
            translated_ar = self.ar + np.array(s, dtype=self.ar.dtype)
            translated_shape = Shape(translated_ar, self.idx)
            if translated_shape.geom_state <= board.geom_state:
                valid_shapes.append(translated_shape)
        return valid_shapes


class Visualizer:

    def visualize(self, shape_set: ShapeSet, board: Shape, basis: np.ndarray, colors: list[str], individually=False):
        """Plot a 3d interactive model of the shape_set on the board."""

        all_shapes = np.vstack([s.ar for s in shape_set])
        x, y, z = np.hsplit(all_shapes, 3)
        # bounding box
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        for i, p in enumerate(shape_set):
            if individually or i == 0:
                ax = self._prepare_axis(xb, yb, zb, board, basis)
            color = colors[p.idx]
            size = 600
            #             elif p.idx==-1:#the board
            #                 color='green'
            #                 size=30
            #             elif p.idx==-2:#the universe
            #                 color='black'
            #                 size=5
            in_orthogonal = p.ar.dot(basis.T)
            ax.scatter(*np.hsplit(in_orthogonal, 3), c=np.array([mpl.colors.to_rgba(color, .7)]), marker='o', s=size,
                       depthshade=False)
        plt.show()

    @staticmethod
    def _prepare_axis(xb, yb, zb, board: Shape, basis: np.ndarray):
        fig = plt.figure()
        fig.set_size_inches(7, 7)
        # Comment or uncomment following both lines to test the fake bounding box:
        ax = fig.add_subplot(111, projection='3d')
        for xb, yb, zb in zip(xb, yb, zb):
            ax.plot([xb], [yb], [zb], 'w')
        # plot the board
        in_orthogonal = board.ar.dot(basis.T)
        ax.scatter(*np.hsplit(in_orthogonal, 3), c='green', marker='o', s=5)
        ax.set_aspect('equal')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        return ax


class Puzzle:
    """ An abstract class representing a puzzle configuration.
        For example, this could be two-dimensional puzzles, or pyramid-like
        puzzles.
        It consists of
        * A list of Pieces defined in the grid space (orthonormal integer grid) and their colors
        * The board, which a special Piece defining legal positions for the balls making up the actual pieces.
        * Legal rotations that will take a piece from a grid back into the grid.
        * The basis mapping the grid space into the real-world space. This is used for 3d interactive visualization.
    """

    def __init__(self):
        ######################
        # intended to be overriden for different types of puzzles
        ######################

        self.pieces, self.colors = self._get_pieces()
        self.board = self._get_board()
        self.symmetries = self._get_symmetries()
        self.basis = self._get_basis()

        ######################
        # computed members below
        ######################

        # for each slot in the board, maps the slot to the left most columns in the state table
        # the keys are string representations of the slot coordinates
        self.board_state_map = {s: i for i, s in enumerate(self.board.geom_state)}

    def solve(self, max_solutions=100) -> list[ShapeSet]:
        state_set = ShapeSet()
        rots = self._rotations(*self.symmetries)
        for p in self.pieces:
            for o in p.orientations(rots, self.basis):
                ts = o.contained_translations(self.board)
                state_set.extend(ts)

        s = self._shape_set_as_state_matrix(state_set).astype(np.int32)
        print("number of oriented, positioned shapes (cover candidates): %d" % len(s))
        sols = []

        def sol_collector(cur_sol):
            sols.append(cur_sol.keys())
            return len(sols) >= max_solutions

        sparse_s = DLMatrix(s.shape[1])
        for r in s:
            sparse_s.add_sparse_row(np.nonzero(r)[0].tolist())

        a = AlgorithmX(sparse_s, sol_collector, True)
        a()
        print("number of solutions: %d" % len(sols))
        sol_sets = []
        for s in sols:
            sol_set = ShapeSet()
            for i in s:
                sol_set.append(state_set[i])
            sol_sets.append(sol_set)
        return sol_sets

    def visualize(self, shape_set: ShapeSet, individually: bool = False):
        """Plot a 3d interactive model of the shape_set on the board."""
        return Visualizer().visualize(shape_set, self.board, self.basis, self.colors, individually)

    @staticmethod
    def _rotations(axes: np.ndarray, folds: np.ndarray) -> np.ndarray:
        """ Generate rotation matrix, by picking a permutation from axes, rotating about each axis in
                the permutation by  360/fold degrees, 0...fold times, and repeating recursively by picking another axis/fold.
            @param axes: Nx3 matrix, with each row an axis in real-world space.
            @param folds: Nx1 vector of integers, corresponding to the axes, so that rotations are done
                in 360/fold degree increments
            @return returns a Nx3x3 array o, where each o[i,:,:] is a rotation matrix.  The matrices will be unique
        """
        fold_lists = []  # a list of lists, with one list per axis, with inner list containing a rot matrix for every fold
        for a, fold in zip(axes, folds):
            g = axis_to_rotation_matrix(a[None, :].T, 2 * math.pi / fold)
            fold_lists.append([g ** p for p in range(fold)])

        rots = []
        for perm in permutations(fold_lists):
            for axisRots in product(*perm):
                cur_rot = np.mat(np.eye(3))
                for r in axisRots:
                    cur_rot *= r
                rots.append(np.around(cur_rot, 10))
        # print ("generated %d rotations"%len(rots))
        return np.unique(np.stack(rots), axis=0)

    def _shape_as_state_vector(self, piece: Shape) -> np.ndarray:
        """
            @return: a binary vector of length (num_board_slots+num_pieces), with a one in ith place, if the jth piece
            occupies the ith  slot on the board, and also a 1 in the num_board_slots+j position, indicating the jth
            piece is used.
        """
        if piece.idx < 0:
            raise ValueError('cannot create state vector for negative index' % piece.idx)
        s = np.zeros(len(self.board_state_map) + len(self.pieces), dtype=np.int8)
        pos_state = [self.board_state_map[v] for v in piece.geom_state]
        s[np.array(pos_state)] = 1
        s[len(self.board_state_map) + piece.idx] = 1
        return s

    def _shape_set_as_state_matrix(self, shape_set: ShapeSet) -> np.ndarray:
        m = np.array([self._shape_as_state_vector(s) for s in shape_set])
        return m

    def _get_board(self) -> Shape:
        """
            @return the board Shape.
        """
        raise NotImplementedError()

    def _get_symmetries(self) -> tuple[np.ndarray, np.ndarray]:
        """
            The symmetries in grid space, specified by rotation axes and folds.  If a piece starts in grid space,
            A rotation around an axis_i  by 1/fold_i  of a circle will place back into the grid.
            The rotations can be composed.  The union of all compositions is all the legal rotations for the puzzle.

            @returns [axes, folds], where axes is Nx3 and folds is Nx1
        """
        raise NotImplementedError()

    @staticmethod
    def _get_pieces() -> tuple[list[Shape], list[str]]:
        """define the pieces of our puzzle.  They are all planar."""
        pieces = [
            # nightstick
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [1, 1]
            ],
            [
                # long uneven corner
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [0, 1]
            ],
            [
                # square
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            [
                # letter p
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
            ],
            [
                # bridge
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [2, 1]
            ],
            [
                # cross
                [1, 0],
                [1, 1],
                [1, 2],
                [0, 1],
                [2, 1],
            ],
            [
                # stairs
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 2],
                [2, 2],
            ],
            [
                # large corner
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
            ],
            [
                # lightning
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
            ],
            [
                # stick
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
            ],
            [
                # short uneven corner
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1]
            ],
            [
                # small corner
                [0, 0],
                [0, 1],
                [1, 0],
            ]
        ]
        for i, p in enumerate(pieces):
            p = np.array(p, dtype=np.int8)
            with_z_axis = np.append(p, np.zeros((p.shape[0], 1), dtype=np.int8), axis=1)
            pieces[i] = Shape(with_z_axis, i)

        # a color for each piece index, to match a real-life puzzle piece color
        colors = (
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

        return pieces, colors

    def _get_basis(self) -> np.ndarray:
        """
            @return the 3x3 matrix, where the columns are the basis of the orthonormal integer grid in
            the real-world space.  Useful for visualization.

            For puzzles boards for which the basis doesn't exist, a different mechanism for visualization will
            be needed.


            The default basis is the identity matrix: I.e the the board slots are on a 3-dimensional integer grid
            in real space.
        """
        return np.eye(3)


class RectanglePuzzle(Puzzle):

    def _get_symmetries(self) -> tuple[np.ndarray, np.ndarray]:
        """
            the Z axis with 4-fold symmetry
            and the Y axis with 2-fold symmetry (flips of the pieces)
            each row is a rotation axis
        """
        return (np.array(
            [[0, 0, 1],
             [0, 1, 0]
             ]),
                np.array([4, 2]).T)

    def _get_board(self) -> Shape:
        """A 5x11 2-dimensional board"""
        board = np.ones((11, 5, 1))
        board = np.transpose(np.array(np.nonzero(board))).astype(np.int8)
        return Shape(board, -1)


class PyramidPuzzle(Puzzle):

    def _get_board(self) -> Shape:
        """
            our board is a pyramid with a 5x5 base, but in our basis,
            one ridge of the pyramid is along the z-axis
        """
        side = 5
        board = np.zeros((side, side, side), dtype=np.int8)
        for z in range(side):
            board[z, :side - z, :side - z] = 1
        board = np.transpose(np.flip(np.nonzero(board), 0).astype(np.int8))
        board = Shape(board, -1)
        return board

    def _get_basis(self) -> np.ndarray:
        return np.array([[1, 0, 0.5],
                         [0, 1, 0.5],
                         [0, 0, 2.0 ** .5 / 2]])

    def _get_symmetries(self) -> tuple[np.ndarray, np.ndarray]:
        """
            diagonals in the X-Y plane, are the rotation axes, with 4-fold symmetry each
            and the Y axis with 2-fold symmetry
            each row is a rotation axis
        """
        return (
            np.array(
                [[1, 1, 0],
                 [1, -1, 0],
                 [0, 1, 0]
                 ]),
            np.array([4, 4, 2]).T)


def axis_to_rotation_matrix(axis, angle_rads):
    """return a rotation matrix to perform a rotation about axis by angle_rads.
       This uses the right-hand rule coordinate system, X pointing right, Y up and Z pointing out of the screen."""
    # could have also used quaternions here

    # create a matrix with axis as one of its basis vectors, and the other two being orthonormal to it
    q, _ = np.linalg.qr(axis, 'complete')
    q = np.mat(q)  # the 1 column is our axis arg

    # the rotation matrix around the x-axis
    co, si = math.cos(angle_rads), math.sin(angle_rads)
    rot = np.matrix(
        [[1.0, 0.0, 0.0],
         [0.0, co, -si],
         [0.0, si, co]])

    # 1) change basis to rename the first column as the X axis, 2) rotate
    # around it, 3) rename the X axis back to be the axis arg
    return np.mat(q * rot * q.I)


if __name__ == '__main__':
    puzzle = PyramidPuzzle()
    # puzzle=RectanglePuzzle()
    solutions = puzzle.solve(10)
    print(solutions)
